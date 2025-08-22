# encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import e3nn
from e3nn import o3
# from e3nn import IrrepsArray # REMOVED
from e3nn.o3 import Linear, TensorProduct, FullyConnectedTensorProduct, Norm, Irreps, SphericalHarmonics
from e3nn.nn import Gate, NormActivation
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, radius_graph
from torch_scatter import scatter_sum, scatter_mean, scatter_softmax
from typing import List, Tuple, Optional


class SafeNormActivation(nn.Module):
    """
    A numerically stable version of e3nn.nn.NormActivation.
    It handles zero-norm features by adding a small epsilon to avoid division by zero.
    """
    def __init__(self, irreps_in, activation=F.silu, eps=1e-8):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.activation = activation
        self.eps = eps
        
        # 预计算每个 irrep 的起始位置，避免重复计算
        self.irreps_slices = []
        start_idx = 0
        for mul, ir in self.irreps_in:
            end_idx = start_idx + mul * ir.dim
            self.irreps_slices.append((start_idx, end_idx, mul, ir))
            start_idx = end_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入检查
        if torch.isnan(x).any():
            print("Warning: Input to SafeNormActivation contains NaN")
            # 将输入中的 NaN 替换为 0
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        
        output = torch.zeros_like(x)
        
        for start_idx, end_idx, mul, ir in self.irreps_slices:
            if ir.l == 0:  # 标量特征
                # 对于标量，直接应用激活函数
                features_slice = x[:, start_idx:end_idx]
                output[:, start_idx:end_idx] = self.activation(features_slice)
            else:  # 向量或更高阶张量特征
                # 重塑为 [batch_size, mul, ir.dim]
                features_slice = x[:, start_idx:end_idx].view(x.size(0), mul, ir.dim)
                
                # 计算每个特征向量的范数
                norms = torch.norm(features_slice, dim=-1, keepdim=True)  # [batch_size, mul, 1]
                
                # 数值稳定性处理
                # 方法1: 使用 clamp 确保最小值
                safe_norms = torch.clamp(norms, min=self.eps)
                
                # 方法2: 对于真正的零向量，保持为零
                zero_mask = (norms < self.eps)
                
                # 归一化
                normalized_features = features_slice / safe_norms
                
                # 应用激活函数到范数
                activated_norms = self.activation(norms.squeeze(-1))  # [batch_size, mul]
                
                # 重新缩放特征
                scaled_features = normalized_features * activated_norms.unsqueeze(-1)
                
                # 对于零向量，输出也应该是零
                scaled_features = torch.where(zero_mask, torch.zeros_like(scaled_features), scaled_features)
                
                # 检查是否仍有 NaN
                if torch.isnan(scaled_features).any():
                    print(f"Warning: NaN detected in irrep {ir} after processing")
                    print(f"Original norms min/max: {norms.min().item():.8f}/{norms.max().item():.8f}")
                    print(f"Zero norm count: {zero_mask.sum().item()}")
                    # 强制清除 NaN
                    scaled_features = torch.where(torch.isnan(scaled_features), 
                                                torch.zeros_like(scaled_features), 
                                                scaled_features)
                
                # 重塑回原始形状并赋值
                output[:, start_idx:end_idx] = scaled_features.view(x.size(0), -1)
        
        # 最终检查
        if torch.isnan(output).any():
            print("Critical Warning: SafeNormActivation still produced NaN, replacing with zeros")
            output = torch.where(torch.isnan(output), torch.zeros_like(output), output)
        
        return output

class RadialMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.SiLU):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor):
        """
        这个 forward pass 增加了一个“安全气囊”，以确保 RadialMLP 内的所有计算
        都以 float32 的精度执行。这可以防止输入的距离信息 x 在转换为 float16
        时发生下溢或精度损失，从而避免后续 Linear 层的梯度计算出现 NaN。
        """
        # 获取当前张量所在的设备类型 (例如 'cuda')
        device_type = x.device.type
        
        # 使用 with 上下文管理器，局部禁用 AMP
        with torch.amp.autocast(device_type=device_type, enabled=False):
            # 1. 将输入 tensor 手动、显式地转换为 float32。
            x_f32 = x.to(torch.float32)
            
            # 2. 在 float32 的“安全区”内执行 MLP 的所有计算。
            output = self.net(x_f32)
            
        return output
    
# def check_clebsch_gordan(ir_in1: o3.Irrep, ir_in2: o3.Irrep, ir_out: o3.Irrep) -> bool:
#     """检查 ir_out 是否包含在 ir_in1 x ir_in2 的张量积中"""
#     l_rule = abs(ir_in1.l - ir_in2.l) <= ir_out.l <= ir_in1.l + ir_in2.l
#     p_rule = ir_out.p == ir_in1.p * ir_in2.p
#     return l_rule and p_rule

class EdgeKeyValueNetwork(nn.Module):
    def __init__(self, irreps_node_input, irreps_sh, irreps_key_output, irreps_value_output, num_basis_radial, radial_mlp_hidden):
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.irreps_key_output = o3.Irreps(irreps_key_output)
        self.irreps_value_output = o3.Irreps(irreps_value_output)
        self.num_basis_radial = num_basis_radial

        self.tp_k = o3.FullyConnectedTensorProduct(
            self.irreps_node_input, self.irreps_sh, self.irreps_key_output,
            shared_weights=False
        )
        self.fc_k = RadialMLP(self.num_basis_radial, radial_mlp_hidden, self.tp_k.weight_numel)

        self.tp_v = o3.FullyConnectedTensorProduct(
            self.irreps_node_input, self.irreps_sh, self.irreps_value_output,
            shared_weights=False
        )
        self.fc_v = RadialMLP(self.num_basis_radial, radial_mlp_hidden, self.tp_v.weight_numel)

    def forward(self, node_features_src: torch.Tensor, edge_sh: torch.Tensor, edge_radial_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k_weights = self.fc_k(edge_radial_emb).to(node_features_src.device)
        k_on_edge = self.tp_k(node_features_src, edge_sh, weight=k_weights)
        v_weights = self.fc_v(edge_radial_emb).to(node_features_src.device)
        v_on_edge = self.tp_v(node_features_src, edge_sh, weight=v_weights)
        return k_on_edge, v_on_edge

class TFNInteractionBlock(nn.Module):
    def __init__(self, irreps_node_input, irreps_node_output, irreps_edge_attr, irreps_sh, num_basis_radial, radial_mlp_hidden=[64], activation_gate=True):
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.num_basis_radial = num_basis_radial

        self.tensor_product = o3.FullyConnectedTensorProduct(
            self.irreps_node_input, self.irreps_sh, self.irreps_node_output,
            shared_weights=False
        )
        self.radial_to_tp_weights = RadialMLP(num_basis_radial, radial_mlp_hidden, self.tensor_product.weight_numel)

        self.linear = Linear(self.irreps_node_output, self.irreps_node_output)

        target_irreps = self.irreps_node_output
        act_norm = F.silu
        self.activation = SafeNormActivation(target_irreps, act_norm)

        if self.irreps_node_input != self.irreps_node_output:
            self.skip_connection_project = Linear(self.irreps_node_input, self.irreps_node_output, internal_weights=True)
        else:
            self.skip_connection_project = None
        # self.activation = nn.Identity()
        # self.skip_connection_project = None

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_sh: torch.Tensor, edge_radial_emb: torch.Tensor,
                edge_length: Optional[torch.Tensor] = None,
                max_radius: Optional[float] = None
                ) -> torch.Tensor:
        # ... (省略 forward 代码, 确保 self.tensor_product 调用时使用 weights=tp_weights) ...
        edge_src, edge_dst = edge_index
        tp_weights = self.radial_to_tp_weights(edge_radial_emb).to(node_features.device)
        node_features_src = node_features[edge_src]
        # 调用 FullTensorProduct 的 forward 时，同样使用 weights 参数
        messages = self.tensor_product(node_features_src, edge_sh, weight=tp_weights)
        aggregated_messages = scatter_sum(messages, edge_dst, dim=0, dim_size=node_features.shape[0])
        transformed_messages = self.linear(aggregated_messages)
        activated_messages = self.activation(transformed_messages)
        if self.skip_connection_project is not None:
            skip_features = self.skip_connection_project(node_features)
        else:
            skip_features = node_features
        updated_node_features = skip_features + activated_messages
        return updated_node_features

        # updated_node_features = activated_messages
        # return updated_node_features
    
# --- SE(3) Transformer Interaction Block (Adjusted for Tensors) ---
class SE3TransformerInteractionBlock(nn.Module):
    def __init__(self, irreps_node_input, irreps_node_output, irreps_edge_attr, irreps_sh, num_basis_radial, radial_mlp_hidden=[64], num_attn_heads=4, fc_neurons=[128, 128], activation_gate=True, use_layer_norm=False):
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.num_basis_radial = num_basis_radial
        self.num_attn_heads = num_attn_heads
        self.use_layer_norm = use_layer_norm

        # --- 1. Define Irreps for Q, K, V per head ---
        assert all(mul % self.num_attn_heads == 0 for mul, ir in self.irreps_node_input), f"Input irreps channels {self.irreps_node_input} must be divisible by num_attn_heads={self.num_attn_heads}"
        self.irreps_value_head = o3.Irreps([ (mul // self.num_attn_heads, ir) for mul, ir in self.irreps_node_input ])
        self.irreps_value = self.irreps_value_head * self.num_attn_heads
        self.irreps_key_query_head = self.irreps_value_head # Using same structure
        self.irreps_key_query = self.irreps_value

        # --- 2. Networks for Q, K, V ---
        self.query_network = Linear(self.irreps_node_input, self.irreps_key_query, internal_weights=True)
        self.kv_network = EdgeKeyValueNetwork(
            irreps_node_input=self.irreps_node_input, irreps_sh=self.irreps_sh,
            irreps_key_output=self.irreps_key_query, irreps_value_output=self.irreps_value,
            num_basis_radial=self.num_basis_radial, radial_mlp_hidden=radial_mlp_hidden
        )

        # --- 3. Attention Mechanism ---
        # Need TP for single head dot product
        self.dot_product_single_head = o3.FullyConnectedTensorProduct(
            self.irreps_key_query_head, self.irreps_key_query_head, "0e"
        )
        self.scale = 1.0 / (self.irreps_key_query_head.dim ** 0.5)

        # --- 4. Output Projection ---
        self.output_projection = Linear(self.irreps_value, self.irreps_node_output, internal_weights=True)

        # --- 5. Feed-Forward Network (FFN) ---
        # Gate initialization logic (already corrected)
        ffn_intermediate_irreps = []
        ffn_act_scalars, ffn_act_gates, ffn_act_features = [], [], []
        ffn_irreps_scalars, ffn_irreps_gated = o3.Irreps(""), o3.Irreps("")
        for mul, ir in self.irreps_node_output: # Base expansion on output irreps
             if ir.l == 0: ffn_intermediate_irreps.append((mul * 2, ir))
             else: ffn_intermediate_irreps.append((mul, ir))
        ffn_intermediate_irreps = o3.Irreps(ffn_intermediate_irreps).simplify()

        target_irreps = ffn_intermediate_irreps
        act_norm = F.silu
        self.ffn_activation = SafeNormActivation(target_irreps, act_norm)

        self.ffn = nn.Sequential(
            Linear(self.irreps_node_output, ffn_intermediate_irreps, internal_weights=True),
            self.ffn_activation,
            Linear(ffn_intermediate_irreps, self.irreps_node_output, internal_weights=True)
        )

        # --- 6. Skip connection projection (if needed) ---
        if self.irreps_node_input != self.irreps_node_output:
            self.skip_projection_attn = Linear(self.irreps_node_input, self.irreps_node_output, internal_weights=True)
        else:
            self.skip_projection_attn = None

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_sh: torch.Tensor, edge_radial_emb: torch.Tensor, edge_length: Optional[torch.Tensor] = None, max_radius: Optional[float] = None) -> torch.Tensor:
        N = node_features.shape[0]
        E = edge_index.shape[1]
        edge_src, edge_dst = edge_index
        H = self.num_attn_heads

        # --- Checkpoint 0: Input Check ---
        if torch.isnan(node_features).any(): raise RuntimeError("NaN detected in input node_features!")
        if torch.isnan(edge_sh).any(): raise RuntimeError("NaN detected in input edge_sh!")
        if torch.isnan(edge_radial_emb).any(): raise RuntimeError("NaN detected in input edge_radial_emb!")

        # --- 1. Calculate Q, K, V ---
        if node_features.dim() == 3:
            node_features_flat = node_features.view(node_features.size(0), -1)
            q = self.query_network(node_features_flat)
        else:
            q = self.query_network(node_features)
        
        # --- Checkpoint 1.1: After Query Projection ---
        if torch.isnan(q).any(): raise RuntimeError("NaN detected in Attention Query (q)!")

        k_on_edge, v_on_edge = self.kv_network(node_features[edge_src], edge_sh, edge_radial_emb)
        
        # --- Checkpoint 1.2: After Key/Value Projection ---
        if torch.isnan(k_on_edge).any(): raise RuntimeError("NaN detected in Attention Key (k_on_edge)!")
        if torch.isnan(v_on_edge).any(): raise RuntimeError("NaN detected in Attention Value (v_on_edge)!")

        # --- 2. Reshape for Multi-Head Attention ---
        q_heads = q.reshape(N, H, self.irreps_key_query_head.dim)
        k_heads = k_on_edge.reshape(E, H, self.irreps_key_query_head.dim)
        v_heads = v_on_edge.reshape(E, H, self.irreps_value_head.dim)

        # --- 3. Calculate Attention Scores ---
        q_heads_on_edge = q_heads[edge_dst]

        attn_logits_list = []
        for h in range(H):
            q_h = q_heads_on_edge[:, h, :]
            k_h = k_heads[:, h, :]
            dot_h = self.dot_product_single_head(q_h, k_h)
            attn_logits_list.append(dot_h)

        attn_logits = torch.cat(attn_logits_list, dim=-1) * self.scale
        
        attn_logits = torch.clamp(attn_logits, -10.0, 10.0)

        # --- Checkpoint 3.1: After Attention Logits Calculation ---
        if torch.isnan(attn_logits).any():
            print(f"DEBUG: Max value in q_heads_on_edge: {q_heads_on_edge.max()}")
            print(f"DEBUG: Max value in k_heads: {k_heads.max()}")
            raise RuntimeError("NaN detected in Attention Logits (attn_logits)!")

        if edge_length is not None and max_radius is not None:
            def soft_unit_step(x, sharpness=10.0): return torch.sigmoid(sharpness * x)
            edge_weight_cutoff = soft_unit_step(1.0 - edge_length.squeeze(-1) / max_radius)
            edge_weight_cutoff = edge_weight_cutoff.unsqueeze(-1).to(attn_logits.device)
            attn_logits = attn_logits * edge_weight_cutoff

        # --- 4. Softmax ---
        attn_weights = scatter_softmax(attn_logits.float(), edge_dst, dim=0)
        
        # --- Checkpoint 4.1: After Softmax ---
        if torch.isnan(attn_weights).any():
            print(f"DEBUG: Max value in attn_logits (input to softmax): {attn_logits.max()}")
            raise RuntimeError("NaN detected after scatter_softmax (attn_weights)!")

        # --- 5. 聚合加权的 Values ---
        weighted_v = v_heads * attn_weights.unsqueeze(-1)
        
        # --- Checkpoint 5.1: After weighting values ---
        if torch.isnan(weighted_v).any(): raise RuntimeError("NaN detected after weighting values (weighted_v)!")

        weighted_v_flat = weighted_v.reshape(E * H, self.irreps_value_head.dim)

        e_indices = torch.arange(E * H, device=edge_dst.device) // H
        h_indices = torch.arange(E * H, device=edge_dst.device) % H
        scatter_index = edge_dst[e_indices] * H + h_indices

        aggregated_v_flat = scatter_sum(weighted_v_flat.float(), scatter_index, dim=0, dim_size=N * H)
        
        # --- Checkpoint 5.2: After scatter_sum aggregation ---
        if torch.isnan(aggregated_v_flat).any(): raise RuntimeError("NaN detected after aggregation (aggregated_v_flat)!")

        aggregated_v = aggregated_v_flat.reshape(N, self.irreps_value.dim)

        # --- 6. Output Projection ---
        projected_output = self.output_projection(aggregated_v)
        
        # --- Checkpoint 6.1: After output projection ---
        if torch.isnan(projected_output).any(): raise RuntimeError("NaN detected after output_projection!")

        # --- 7. First Residual Connection ---
        if self.skip_projection_attn is not None:
            residual_input = self.skip_projection_attn(node_features)
        else:
            assert self.irreps_node_input == self.irreps_node_output, "Input/Output irreps mismatch without skip projection!"
            residual_input = node_features
        
        attn_block_output = residual_input + projected_output
        # --- Checkpoint 7.1: After first residual connection ---
        if torch.isnan(attn_block_output).any(): raise RuntimeError("NaN detected after first residual connection!")

        # print("\n--- FFN INPUT DEBUG ---")
        # print(f"Shape of attn_block_output: {attn_block_output.shape}")
        # print(f"Dtype of attn_block_output: {attn_block_output.dtype}")
        # print(f"Max value in attn_block_output: {torch.max(torch.abs(attn_block_output))}")
        # print(f"Min value in attn_block_output: {torch.min(attn_block_output)}")
        # print(f"Mean value in attn_block_output: {torch.mean(attn_block_output)}")

        # Step 8.1: First Linear Layer
        ffn_linear1_out = self.ffn[0](attn_block_output)
        if torch.isnan(ffn_linear1_out).any():
            print("--- FFN DEBUG: NaN after linear1 ---")
            print(f"Max value in attn_block_output (input): {torch.max(torch.abs(attn_block_output))}")
            raise RuntimeError("NaN detected inside FFN, after the first Linear layer!")

        # Step 8.2: Activation Layer (NormActivation)
        ffn_activation_out = self.ffn[1](ffn_linear1_out)
        if torch.isnan(ffn_activation_out).any():
            print("--- FFN DEBUG: NaN after activation ---")
            print(f"Max value in ffn_linear1_out (input to activation): {torch.max(torch.abs(ffn_linear1_out))}")
            # Let's inspect the norm calculation inside NormActivation
            # NormActivation(x) = activation(norm(x)) * x / norm(x)
            # NaN can happen if norm(x) is zero.
            norms = o3.Norm(self.ffn[1].irreps_in)(ffn_linear1_out) # Manually compute the norm
            zero_norms = torch.where(norms == 0)[0]
            if len(zero_norms) > 0:
                print(f"Detected {len(zero_norms)} zero-norm features going into NormActivation at indices: {zero_norms}")
            raise RuntimeError("NaN detected inside FFN, after the NormActivation layer!")

        # Step 8.3: Second Linear Layer
        ffn_output = self.ffn[2](ffn_activation_out)
        if torch.isnan(ffn_output).any():
            print("--- FFN DEBUG: NaN after linear2 ---")
            print(f"Max value in ffn_activation_out (input to linear2): {torch.max(torch.abs(ffn_activation_out))}")
            raise RuntimeError("NaN detected inside FFN, after the second Linear layer!")

        # --- 9. Second Residual Connection ---
        final_output = attn_block_output + ffn_output
        
        # --- Checkpoint 9.1: After second residual connection ---
        if torch.isnan(final_output).any(): raise RuntimeError("NaN detected after second residual connection!")

        return final_output

# --- 主等变 GNN 编码器 (SE3InvariantGraphEncoder - Adjusted for Tensors) ---
# In SE3InvariantGraphEncoder.__init__
class SE3InvariantGraphEncoder(nn.Module):
    def __init__(self, num_atom_types: int,           
                 embedding_dim_scalar: int,
                 # actual_atom_feature_dim_from_data: int = 119,
                 irreps_node_hidden: str = "64x0e + 16x1o + 8x2e", 
                 irreps_node_output: str = "128x0e", # This is the Irreps of the equivariant output
                 irreps_edge_attr: str = "0x0e", 
                 irreps_sh: str = "1x0e + 1x1o + 1x2e", 
                 max_radius: float = 5.0, num_basis_radial: int = 16, 
                 radial_mlp_hidden: list = [64, 64], 
                 num_interaction_layers: int = 3, 
                 num_attn_heads: int = 2, 
                 use_attention: bool = True, 
                 activation_gate: bool = True,): # activation_gate might be unused if FFN structure changed
        super().__init__()
        self.eps = 1e-8
        self.irreps_node_input = o3.Irreps(f"{embedding_dim_scalar}x0e")

        self.node_embedding = nn.Embedding(num_atom_types, embedding_dim_scalar)

        # self.lin = nn.Linear(actual_atom_feature_dim_from_data, embedding_dim_scalar)
        
        # Store target Irreps objects
        self.irreps_node_hidden_obj = o3.Irreps(irreps_node_hidden)
        self.irreps_node_output_obj = o3.Irreps(irreps_node_output) # The Irreps of the final equivariant features

        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.max_radius = max_radius
        self.num_basis_radial = num_basis_radial
        self.use_attention = use_attention

        self.spherical_harmonics = SphericalHarmonics(self.irreps_sh, normalize=True, normalization='component')
        
        radial_input_dim = 1 # For edge_dist
        edge_attr_irreps = o3.Irreps(irreps_edge_attr)
        if edge_attr_irreps.dim > 0: 
            radial_input_dim += edge_attr_irreps.dim

        print(f"DEBUG: radial_input_dim = {radial_input_dim}")
        print(f"DEBUG: edge_attr_irreps = {edge_attr_irreps}, dim = {edge_attr_irreps.dim}")
        
        # 使用正确的输入维度创建 RadialMLP
        self.radial_embedding = RadialMLP(
            input_dim=radial_input_dim,  # 明确指定参数名
            hidden_dims=radial_mlp_hidden, 
            output_dim=self.num_basis_radial
        )

        self.interaction_layers_module = nn.ModuleList()
        current_input_irreps_for_block = self.irreps_node_input # Start with initial scalar embeddings

        for i in range(num_interaction_layers):
            # Determine output irreps for this layer
            if i < num_interaction_layers - 1:
                current_output_irreps_for_block = self.irreps_node_hidden_obj
            else: # Last interaction layer
                current_output_irreps_for_block = self.irreps_node_output_obj # Will output these equivariant features

            if use_attention:
                 block = SE3TransformerInteractionBlock(
                     irreps_node_input=str(current_input_irreps_for_block), 
                     irreps_node_output=str(current_output_irreps_for_block),
                     irreps_edge_attr=str(self.irreps_edge_attr),     # <--- 添加
                     irreps_sh=str(self.irreps_sh),                 # <--- 添加
                     num_basis_radial=self.num_basis_radial,        # <--- 添加
                     radial_mlp_hidden=radial_mlp_hidden, # Make sure this is passed if SE3TInteractionBlock needs it
                     num_attn_heads=num_attn_heads,
                     # activation_gate was removed as unused from my SE3T block suggestion, 
                     # add it back if your SE3T block definition requires it.
                 )
            else: # TFNInteractionBlock
                 block = TFNInteractionBlock(
                     irreps_node_input=str(current_input_irreps_for_block), 
                     irreps_node_output=str(current_output_irreps_for_block),
                     irreps_edge_attr=str(self.irreps_edge_attr),     # <--- 添加
                     irreps_sh=str(self.irreps_sh),                 # <--- 添加
                     num_basis_radial=self.num_basis_radial,        # <--- 添加
                     radial_mlp_hidden=radial_mlp_hidden # Make sure this is passed if TFNInteractionBlock needs it
                     # activation_gate was removed as unused from my TFN block suggestion,
                     # add it back if your TFN block definition requires it.
                 )
            self.interaction_layers_module.append(block) 
            current_input_irreps_for_block = current_output_irreps_for_block
        
        # No final_invariant_projection_layers or final_scalar_dim needed here
        # if the forward method solely returns equivariant features and their irreps.

    def forward(self, data: Data) -> Tuple[torch.Tensor, o3.Irreps]:
        # --- 1. Initial Data Extraction and Feature Projection ---
        pos = data.pos.float()
        node_features_indices = data.x

        if torch.isnan(pos).any():
            raise RuntimeError("NaN detected in input data.pos!")

        edge_index = data.edge_index
        edge_src, edge_dst = edge_index

        current_node_features = self.node_embedding(node_features_indices)
        current_node_irreps = self.irreps_node_input

        # --- Checkpoint 1.2: After initial linear projection ---
        if torch.isnan(current_node_features).any():
            raise RuntimeError("NaN detected after the initial nn.Linear layer (self.lin)!")

        # --- 2. Compute Edge Vectors (with Periodic Boundary Conditions if available) ---
        if hasattr(data, 'lattice') and data.lattice is not None and \
           hasattr(data, 'edge_shift') and data.edge_shift is not None:
            
            if hasattr(data, 'batch') and data.batch is not None:
                batch_idx_nodes = data.batch
            else:
                batch_idx_nodes = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)

            if data.lattice.shape[0] == 1:
                lattice_for_edges = data.lattice.expand(edge_src.shape[0], 3, 3)
            else:
                lattice_for_edges = data.lattice[batch_idx_nodes[edge_src]]
            
            shift_vectors_cartesian = torch.einsum('ei,eij->ej', data.edge_shift.to(pos.dtype), lattice_for_edges)
            edge_vec = (pos[edge_dst] - pos[edge_src] + shift_vectors_cartesian)
        else:
            edge_vec = pos[edge_dst] - pos[edge_src]

        # --- Checkpoint 2.1: After edge vector calculation ---
        if torch.isnan(edge_vec).any():
            raise RuntimeError("NaN detected after edge vector calculation!")

        # --- 3. Compute Geometric Edge Attributes ---
        edge_dist = torch.norm(edge_vec, dim=-1, keepdim=True)
        
        edge_vec_normalized = torch.zeros_like(edge_vec)
        valid_edge_mask = edge_dist.squeeze(-1) > self.eps
        if valid_edge_mask.any():
            edge_vec_normalized[valid_edge_mask] = edge_vec[valid_edge_mask] / edge_dist[valid_edge_mask].clamp(min=self.eps)

        # --- Checkpoint 3.1: After normalization ---
        if torch.isnan(edge_vec_normalized).any():
            # This often happens if edge_dist is zero for some edges, despite the clamp.
            problematic_edges = torch.where(torch.isnan(edge_vec_normalized).any(dim=1))[0]
            print(f"NaN in edge_vec_normalized at edge indices: {problematic_edges}")
            print(f"Original edge_vec for these edges:\n{edge_vec[problematic_edges]}")
            print(f"Original edge_dist for these edges:\n{edge_dist[problematic_edges]}")
            raise RuntimeError("NaN detected after edge vector normalization!")

        edge_sh = self.spherical_harmonics(edge_vec_normalized)
        
        radial_input = edge_dist
        if hasattr(data, 'edge_attr') and data.edge_attr is not None and self.irreps_edge_attr.dim > 0:
            radial_input = torch.cat([edge_dist, data.edge_attr.to(pos.dtype)], dim=-1)
        
        edge_radial_emb = self.radial_embedding(radial_input)

        # --- Checkpoint 3.2: After edge attribute calculation ---
        if torch.isnan(edge_sh).any():
            raise RuntimeError("NaN detected after spherical harmonics (edge_sh)!")
        if torch.isnan(edge_radial_emb).any():
            raise RuntimeError("NaN detected after radial embedding (edge_radial_emb)!")

        # --- 4. Propagate Features through Interaction Layers ---
        for i, interaction_block in enumerate(self.interaction_layers_module):
            current_node_features = interaction_block(
                node_features=current_node_features,
                edge_index=edge_index,
                edge_sh=edge_sh,
                edge_radial_emb=edge_radial_emb,
                edge_length=edge_dist,
                max_radius=self.max_radius
            )
            # --- Checkpoint 4.1: Inside the loop, after each block ---
            if torch.isnan(current_node_features).any():
                raise RuntimeError(f"NaN detected after interaction block number {i}!")
                
            current_node_irreps = o3.Irreps(interaction_block.irreps_node_output)

            # current_node_features = torch.tanh(current_node_features)

        return current_node_features, current_node_irreps
