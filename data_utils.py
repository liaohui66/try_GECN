# data_utils.py
import torch
import json
import ast 
import numpy as np
import pandas as pd 
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.neighborlist import neighbor_list
from ase.atoms import Atom
from ase.data import atomic_numbers
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder
from jarvis.core.specie import Specie 
from e3nn import o3 
from tqdm import tqdm
from typing import Optional, List, Dict, Any, Tuple
import torch_geometric.transforms as T


# --- 1. 原子特征编码器 ---
_ATOM_FEATURES_LOOKUP_TABLE: Optional[np.ndarray] = None
_ATOM_FEATURE_DIM: int = -1
_ATOM_ENCODER_INSTANCE: Optional[OneHotEncoder] = None 

# 1. 替换 initialize_atom_features 函数
def initialize_atom_features(atom_feat_config: Optional[Dict] = None,
                             force_reinit: bool = False
                            ) -> Tuple[np.ndarray, int]:
    """
    一个简化的、绝对稳定的原子特征初始化版本 (最终修复版)。
    它不使用 magpie 或 OneHotEncoder。
    data.x 将只包含原子序数本身，供模型内部的 nn.Embedding 使用。
    """
    global _ATOM_FEATURES_LOOKUP_TABLE, _ATOM_FEATURE_DIM

    # 这个函数现在变得非常简单
    print("Initializing atom features using a SIMPLE & STABLE method (atomic number).")
    
    max_elements = atom_feat_config.get("max_elements", 101) if atom_feat_config else 101
    
    # 特征维度现在是 1，只包含原子序数。
    _ATOM_FEATURE_DIM = 1
    
    # 查找表是一个 [101, 1] 的矩阵，值为 [[1], [2], [3], ..., [101]]。
    # 模型将使用这些值作为 nn.Embedding 的输入索引。
    _ATOM_FEATURES_LOOKUP_TABLE = np.arange(1, max_elements + 1).reshape(-1, 1).astype(np.float32)

    print(f"Atom features initialized. Lookup table shape: {_ATOM_FEATURES_LOOKUP_TABLE.shape}, "
          f"Feature dimension: {_ATOM_FEATURE_DIM}")
          
    return _ATOM_FEATURES_LOOKUP_TABLE, _ATOM_FEATURE_DIM


# --- 2. Helper function for radial cutoff (from EATGNN) ---
def r_cut2D(radial_cutoff_base: float, ase_atoms_obj: Atom) -> float: # Changed 'cell' to 'ase_atoms_obj'
    # This function was already good from your EATGNN script.
    # structure=AseAtomsAdaptor.get_structure(ase_atoms_obj) # Not needed, ase_atoms_obj is ASE Atoms
    cell_matrix = ase_atoms_obj.get_cell(complete=True).array
    if np.all(np.abs(cell_matrix) < 1e-6): # Handle non-periodic systems (molecules)
        return radial_cutoff_base
    
    norms = [np.linalg.norm(cell_matrix[i]) for i in range(3) if np.any(np.abs(cell_matrix[i]) > 1e-6)]
    if not norms: # All cell vectors are zero (e.g. molecule but cell was [0,0,0])
        return radial_cutoff_base
        
    r_cut = max(max(norms), radial_cutoff_base)
    # r_cut=min(r_cut,max_allowable_radius) # Optional cap from EATGNN
    return r_cut

_print_counter = 0
# --- 3. Core function to process one structure and its target ---
def create_pyg_data(
    pymatgen_structure: Structure,
    piezo_tensor_target: Any, # Should be convertible to 3x3x3 tensor
    atom_features_lookup: np.ndarray, # The global `fea` table
    atom_feature_dim: int,            # The global `dim`
    radial_cutoff: float,
    dtype: torch.dtype,
    irreps_edge_attr_str: Optional[str] = "0x0e" # For creating data.edge_attr if needed by model
) -> Data:
    """
    Converts a Pymatgen Structure and its piezo tensor target into a PyG Data object.
    This function incorporates logic from EATGNN's `datatransform`.
    """
    # <<< 引用并修改全局计数器 >>>
    global _print_counter

    ase_atoms = AseAtomsAdaptor.get_atoms(pymatgen_structure)

    # --- 关键修改：Node features (data.x) ---
    atomic_nums = [atomic_numbers[s] for s in ase_atoms.get_chemical_symbols()]
    # x 直接就是原子序数列表，形状为 [num_atoms]，类型为 long
    x = torch.tensor(atomic_nums, dtype=torch.long)

    # Node positions (data.pos)
    pos = torch.tensor(ase_atoms.get_positions(), dtype=dtype)

    # Lattice (data.lattice)
    lattice = torch.tensor(ase_atoms.get_cell(complete=True).array, dtype=dtype).unsqueeze(0)

    # Edges and shifts (data.edge_index, data.edge_shift)
    effective_cutoff = r_cut2D(radial_cutoff, ase_atoms)
    if len(ase_atoms) > 1:
        ase_atoms_for_nl = ase_atoms.copy()
        ase_atoms_for_nl.set_pbc(True)
        
        edge_src, edge_dst, edge_shift_raw = neighbor_list(
            "ijS", a=ase_atoms_for_nl, cutoff=effective_cutoff, self_interaction=False
        )
        edge_index = torch.stack([
            torch.from_numpy(edge_src), torch.from_numpy(edge_dst)
        ], dim=0).long()
        edge_shift = torch.from_numpy(edge_shift_raw).to(dtype=dtype)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_shift = torch.empty((0, 3), dtype=dtype)
        
    # Target piezoelectric tensor (data.y_piezo)
    y_piezo = torch.tensor(piezo_tensor_target, dtype=dtype)
    if y_piezo.shape != (3, 3, 3):
        if y_piezo.numel() == 27:
            y_piezo = y_piezo.reshape(3, 3, 3)
        else:
            raise ValueError(f"Piezo tensor target from JSON should be 3x3x3 or reshapeable, but got {y_piezo.shape}")

    # <<< 新增：在这里添加诊断打印 >>>
    if _print_counter < 5: # 只打印前5个样本的 y_piezo 信息
        print(f"\n--- [DEBUG] Sample {_print_counter + 1} Target (y_piezo) Stats ---")
        print(f"Shape: {y_piezo.shape}")
        # 使用 .item() 将tensor标量转换为python数字，避免打印设备信息
        print(f"Mean: {y_piezo.mean().item():.4e}")
        print(f"Std Dev: {y_piezo.std().item():.4e}")
        print(f"Max Value: {y_piezo.max().item():.4e}")
        print(f"Min Value: {y_piezo.min().item():.4e}")
        print("--------------------------------------------------")
        _print_counter += 1

    # Optional: edge_attr
    data_edge_attr = None
    if irreps_edge_attr_str and irreps_edge_attr_str != "0x0e":
        edge_attr_dim = o3.Irreps(irreps_edge_attr_str).dim
        if edge_attr_dim > 0:
            if edge_index.shape[1] > 0:
                data_edge_attr = torch.zeros(edge_index.shape[1], edge_attr_dim, dtype=dtype)
            else:
                data_edge_attr = torch.empty((0, edge_attr_dim), dtype=dtype)

    # Assemble Data object
    data_params = {
        'x': x, 'pos': pos, 'edge_index': edge_index,
        'lattice': lattice, 'edge_shift': edge_shift,
        'y_piezo': y_piezo
    }
    if data_edge_attr is not None:
        data_params['edge_attr'] = data_edge_attr
        
    pyg_data = Data(**data_params)
    return pyg_data

# --- 4. Functions to parse your special concatenated file ---
def parse_concatenated_file(
    file_path: str,
    structure_end_line: int,
    piezo_start_line: int,
    piezo_end_line: int
) -> Tuple[List[Dict[str, Any]], List[Any]]: # 更精确的类型提示
    """
    Parses the special concatenated file format for structures and piezo tensors.
    """
    structures_as_dicts: List[Dict[str, Any]] = []
    piezo_tensors_as_lists: List[Any] = [] # 通常是 List[List[List[float]]]

    print(f"Reading concatenated file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}")
        return [], [] # Return empty lists if file not found
    except Exception as e:
        print(f"ERROR: Could not read file {file_path}. Error: {e}")
        return [], []


    # Parse Pymatgen Structure JSON objects
    print(f"Parsing structures (lines 1 to {structure_end_line})...")
    current_object_str_buffer = ""
    brace_counter = 0
    in_object_currently = False # Track if we are inside a { } block

    for i in range(min(len(lines), structure_end_line)): # Ensure we don't go out of bounds
        line = lines[i]
        
        # Heuristic to find start of a new JSON object if not already in one
        if not in_object_currently and line.strip().startswith("{"):
            in_object_currently = True
            current_object_str_buffer = line # Start new buffer with this line
            brace_counter = line.count("{") - line.count("}")
            if brace_counter == 0 and current_object_str_buffer.strip().endswith("}"): # Single line object
                 # Process immediately
                 try:
                    obj_to_parse = current_object_str_buffer.strip()
                    if obj_to_parse.endswith(','): obj_to_parse = obj_to_parse[:-1]
                    structures_as_dicts.append(json.loads(obj_to_parse))
                 except json.JSONDecodeError as e:
                    print(f"Error decoding single-line structure JSON near line {i+1}: {e}")
                 current_object_str_buffer = ""
                 in_object_currently = False
            continue # Move to next line

        if in_object_currently:
            current_object_str_buffer += line
            brace_counter += line.count("{")
            brace_counter -= line.count("}")
            
            if brace_counter == 0: # End of a multi-line object
                try:
                    obj_to_parse = current_object_str_buffer.strip()
                    if obj_to_parse.endswith(','): obj_to_parse = obj_to_parse[:-1]
                    structures_as_dicts.append(json.loads(obj_to_parse))
                except json.JSONDecodeError as e:
                    print(f"Error decoding multi-line structure JSON object ending near line {i+1}: {e}")
                    # print(f"Problematic string part for structure: {current_object_str_buffer[:500]}...")
                current_object_str_buffer = ""
                in_object_currently = False # Reset for next potential object

    # Parse Piezoelectric Tensor list-like strings
    print(f"Parsing piezo tensors (lines {piezo_start_line} to {piezo_end_line})...")
    current_tensor_str_buffer = ""
    bracket_counter = 0
    in_tensor_currently = False # Track if we are inside a [ ] block

    # piezo_start_line is 1-based, lines is 0-based
    for i in range(min(len(lines), piezo_start_line - 1), min(len(lines), piezo_end_line)):
        line = lines[i]

        if not in_tensor_currently and line.strip().startswith("["):
            in_tensor_currently = True
            current_tensor_str_buffer = line
            bracket_counter = line.count("[") - line.count("]")
            if bracket_counter == 0 and current_tensor_str_buffer.strip().endswith("]"): # Single line tensor
                try:
                    tensor_to_eval = current_tensor_str_buffer.strip()
                    if tensor_to_eval.endswith(','): tensor_to_eval = tensor_to_eval[:-1]
                    piezo_tensors_as_lists.append(ast.literal_eval(tensor_to_eval))
                except Exception as e:
                    print(f"Error evaluating single-line piezo tensor near line {i+1}: {e}")
                current_tensor_str_buffer = ""
                in_tensor_currently = False
            continue

        if in_tensor_currently:
            current_tensor_str_buffer += line
            bracket_counter += line.count("[")
            bracket_counter -= line.count("]")

            if bracket_counter == 0: # End of a multi-line tensor
                try:
                    tensor_to_eval = current_tensor_str_buffer.strip()
                    if tensor_to_eval.endswith(','): tensor_to_eval = tensor_to_eval[:-1]
                    piezo_tensors_as_lists.append(ast.literal_eval(tensor_to_eval))
                except Exception as e:
                    print(f"Error evaluating multi-line piezo tensor string ending near line {i+1}: {e}")
                    # print(f"Problematic string part for piezo: {current_tensor_str_buffer[:500]}...")
                current_tensor_str_buffer = ""
                in_tensor_currently = False

    if len(structures_as_dicts) != len(piezo_tensors_as_lists):
        print(f"Warning: Mismatch in parsed structures ({len(structures_as_dicts)}) and piezo tensors ({len(piezo_tensors_as_lists)}).")
    
    return structures_as_dicts, piezo_tensors_as_lists


# --- 5. Main function to load the dataset (ENTRY POINT for other scripts) ---
def load_piezo_dataset( # 这个函数名是 train.py 正在调用的
    large_json_file_path: str,         # train.py 会传递 CONCATENATED_JSON_FILE
    structure_key_in_json: str,    # train.py 会传递 "structure" (或你确认的键名)
    piezo_tensor_key_in_json: str, # train.py 会传递 "total" (或你确认的键名)
    radial_cutoff: float,
    device: torch.device,
    dtype: torch.dtype,
    atom_feat_config: Optional[Dict] = None,
    irreps_edge_attr_for_data: Optional[str] = "0x0e",
    limit_n: Optional[int] = None
) -> List[Data]:
    # 原子特征初始化 (确保它被正确调用并使用 atom_feat_config)
    afe_params = atom_feat_config if atom_feat_config else {}
    atom_features_table, atom_feat_dim = initialize_atom_features(atom_feat_config=afe_params) # 确保这里传递的是字典
    if atom_features_table is None or atom_feat_dim == -1:
        raise RuntimeError("Atom features could not be initialized properly.")

    print(f"Loading large JSON file: {large_json_file_path}")
    data_dict = None
    try:
        with open(large_json_file_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f) # 加载整个文件为一个大字典
    except json.JSONDecodeError as e:
        print(f"FATAL: Could not decode the main JSON file '{large_json_file_path}'. Error: {e}")
        try: # 尝试打印文件开头帮助调试
            with open(large_json_file_path, 'r', encoding='utf-8') as f_err:
                print("\n--- Start of file (first ~500 chars) that caused error ---")
                print(f_err.read(500)); print("--- End of sample content ---")
        except: pass
        return []
    except Exception as e_open:
        print(f"FATAL: Could not open or read main JSON file '{large_json_file_path}'. Error: {e_open}")
        return []

    if not isinstance(data_dict, dict):
        print(f"ERROR: Loaded data from '{large_json_file_path}' is not a dictionary (type: {type(data_dict)}).")
        return []

    # 检查键是否存在
    if structure_key_in_json not in data_dict:
        raise KeyError(f"Structure key '{structure_key_in_json}' not found in JSON. Available keys: {list(data_dict.keys())}")
    if piezo_tensor_key_in_json not in data_dict:
        raise KeyError(f"Piezo tensor key '{piezo_tensor_key_in_json}' not found in JSON. Available keys: {list(data_dict.keys())}")

    structure_dicts_list = data_dict[structure_key_in_json]
    piezo_tensors_list = data_dict[piezo_tensor_key_in_json]

    if not isinstance(structure_dicts_list, list):
        raise TypeError(f"Data under key '{structure_key_in_json}' is not a list (is {type(structure_dicts_list)}).")
    if not isinstance(piezo_tensors_list, list):
        raise TypeError(f"Data under key '{piezo_tensor_key_in_json}' is not a list (is {type(piezo_tensors_list)}).")

    # 打印加载到的数量，用于调试
    print(f"Found {len(structure_dicts_list)} structures under key '{structure_key_in_json}'.")
    print(f"Found {len(piezo_tensors_list)} piezo tensors under key '{piezo_tensor_key_in_json}'.")

    if len(structure_dicts_list) != len(piezo_tensors_list):
        min_len = min(len(structure_dicts_list), len(piezo_tensors_list))
        print(f"Warning: Mismatch in number of structures ({len(structure_dicts_list)}) "
              f"and piezo tensors ({len(piezo_tensors_list)}).")
        if limit_n is None and min_len == 0 : # 如果没有限制且最短为0，则是一个严重问题
             raise ValueError("Data mismatch and no limit_n specified, and one list is empty. Please check the JSON keys or file content.")
        print(f"Will process up to the minimum available {min_len} pairs or limit_n.")
        num_entries_to_process = min_len
    else:
        num_entries_to_process = len(structure_dicts_list)

    if limit_n is not None and limit_n >= 0:
        num_entries_to_process = min(num_entries_to_process, limit_n)
    
    if num_entries_to_process == 0:
        print("No entries to process from the loaded JSON data.")
        return []

    dataset = []
    print(f"Converting {num_entries_to_process} entries to PyG Data objects...")
    for i in tqdm(range(num_entries_to_process)):
        struct_dict = structure_dicts_list[i]
        piezo_data = piezo_tensors_list[i]
        try:
            if not isinstance(struct_dict, dict):
                raise TypeError(f"Structure entry {i} is not a dict, but {type(struct_dict)}")
            pymatgen_structure = Structure.from_dict(struct_dict)
            pyg_data_obj = create_pyg_data(
                pymatgen_structure, piezo_data,
                atom_features_table, atom_feat_dim,
                radial_cutoff, dtype,
                irreps_edge_attr_for_data
            )
            pyg_data_obj.original_index = i
            dataset.append(pyg_data_obj)
        except Exception as e:
            mat_id = "Unknown"
            if isinstance(struct_dict, dict): mat_id = struct_dict.get('material_id', f'entry_index_{i}')
            print(f"Error creating Data object for entry {i} (ID: {mat_id}): {e}")
            # import traceback; traceback.print_exc() # Uncomment for detailed error
    return dataset


class RandomRotatePiezo(T.BaseTransform):
    """
    一个 PyG 变换，用于对节点位置和三阶压电张量目标应用随机旋转。
    修正版本：确保了张量数据类型正确，并使用 einsum 进行清晰、正确的张量旋转。
    """
    def __call__(self, data):
        # 1. 生成一个随机的3x3旋转矩阵 R
        
        # --- 修正 1: 确保所有用于计算的量都是 Tensor ---
        # 原始代码中的 .item() 会将张量变为Python浮点数，导致 torch.sin() 报错。
        # 我们移除 .item()，让 angle 保持为张量。
        
        # 生成一个随机角度（作为标量张量）
        angle = torch.rand(1, device=data.pos.device, dtype=data.pos.dtype) * 2 * torch.pi
        
        # 生成一个随机旋转轴
        axis = torch.randn(3, device=data.pos.device, dtype=data.pos.dtype)
        axis = axis / torch.norm(axis)

        # 使用罗德里格斯旋转公式 (Rodrigues' rotation formula)
        # K 是旋转轴的反对称矩阵
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], device=data.pos.device, dtype=data.pos.dtype)
        
        I = torch.eye(3, device=data.pos.device, dtype=data.pos.dtype)
        
        # 由于 angle 是张量，这里的 torch.sin 和 torch.cos 可以正常工作
        R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.matmul(K, K)

        # 2. 旋转原子坐标 (vector)
        # 新坐标 = R * 旧坐标 (矩阵乘法)
        # 由于坐标是行向量，所以乘以 R 的转置
        data.pos = torch.matmul(data.pos, R.T)
        
        # 3. 旋转晶格参数 (如果存在)
        if hasattr(data, 'lattice') and data.lattice is not None:
            # lattice 是 (1, 3, 3) 或 (3, 3) 的张量
            if data.lattice.ndim >= 2:
                # 晶格向量也是行向量，所以乘以 R 的转置
                data.lattice = torch.matmul(data.lattice, R.T)
        
        # 4. 旋转三阶压电张量 (3rd-rank tensor)
        if hasattr(data, 'y_piezo') and data.y_piezo is not None:
            y = data.y_piezo
            # 确保 y 有一个批次维度，以方便 einsum 操作
            if y.ndim == 3:
                y = y.unsqueeze(0)  # 形状变为 [1, 3, 3, 3]

            # --- 修正 2: 使用 torch.einsum 进行清晰且正确的张量旋转 ---
            # 旋转规则: T'_{pqr} = R_pi * R_qj * R_rk * T_ijk
            # 'b' 代表批次维度, 'ijk' 是原始坐标系, 'pqr' 是新坐标系。
            y_rotated = torch.einsum('pi,qj,rk,bijk->bpqr', R, R, R, y)
            
            # 如果原始输入没有批次维度，就将其移除
            if data.y_piezo.ndim == 3:
                data.y_piezo = y_rotated.squeeze(0)
            else:
                data.y_piezo = y_rotated
                
        return data
    
class PiezoTensorScaler:
    """
    A scaler for standardizing 3x3x3 piezoelectric tensors component-wise.
    It computes the mean and std deviation for each of the 27 components
    from a training dataset and applies the transformation.
    """
    def __init__(self, epsilon=1e-8):
        # self.mean 和 self.std 将会是 3x3x3 的张量
        self.mean = None
        self.std = None
        self.epsilon = epsilon # 用于防止除以零

    def fit(self, dataset: list):
        """
        Computes the component-wise mean and std of the y_piezo attribute 
        from a list of Data objects.
        
        Args:
            dataset (list): A list of torch_geometric.data.Data objects (the training set).
        """
        if not dataset:
            raise ValueError("Cannot fit scaler on an empty dataset.")
            
        print("Fitting PiezoTensorScaler on the training dataset (component-wise)...")
        
        # 1. 将所有训练样本的 y_piezo 张量堆叠起来
        # 形成一个形状为 [num_samples, 3, 3, 3] 的大张量
        all_y_piezo = torch.stack([data.y_piezo for data in dataset])
        
        # 2. 沿着样本维度 (dim=0) 计算逐分量的均值和标准差
        self.mean = torch.mean(all_y_piezo, dim=0)
        self.std = torch.std(all_y_piezo, dim=0)
        
        # 3. 为标准差中接近于零的值添加一个小的 epsilon，防止在 transform 中除以零
        # 这在某些分量方差极小或为零时非常重要
        self.std[torch.abs(self.std) < self.epsilon] = 1.0
        
        print("Scaler fitted successfully.")
        print(f"  - Mean tensor shape: {self.mean.shape}")
        print(f"  - Std tensor shape: {self.std.shape}")
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Applies component-wise standardization.
        PyTorch's broadcasting handles the element-wise operation correctly.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted yet. Call .fit(dataset) first.")
        
        # 将 scaler 的 mean 和 std 移动到与输入数据相同的设备
        mean_on_device = self.mean.to(data.device)
        std_on_device = self.std.to(data.device)

        # data shape: [..., 3, 3, 3]
        # mean_on_device shape: [3, 3, 3]
        # std_on_device shape: [3, 3, 3]
        # 广播机制会自动对齐维度并进行逐元素操作
        return (data - mean_on_device.unsqueeze(0)) / std_on_device.unsqueeze(0)

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Applies the inverse component-wise transformation.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted yet. Call .fit(dataset) first.")

        mean_on_device = self.mean.to(data.device)
        std_on_device = self.std.to(data.device)
            
        return data * std_on_device.unsqueeze(0) + mean_on_device.unsqueeze(0)
        
    def save(self, filepath: str):

        if self.mean is None or self.std is None:
            print("Warning: Trying to save an unfitted scaler.")
            return
        torch.save({'mean': self.mean, 'std': self.std}, filepath)
        print(f"Scaler saved to {filepath}")

    def load(self, filepath: str):

        state = torch.load(filepath)
        self.mean = state['mean']
        self.std = state['std']
        print(f"Scaler loaded from {filepath}")
        return self