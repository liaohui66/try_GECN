import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import os
import logging
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_utils import load_piezo_dataset, initialize_atom_features, RandomRotatePiezo, PiezoTensorScaler
try:
    # 优先尝试绝对导入
    from model import EndToEndPiezoNet
    from debug_utils import DummyEncoder, DummyGroupConvLayer, DummyIterativeBlock
except ImportError:
    # 如果失败，再尝试相对导入
    from .model import EndToEndPiezoNet
    from .debug_utils import DummyEncoder, DummyGroupConvLayer, DummyIterativeBlock
from torch_geometric.transforms import Compose

def setup_for_distributed(is_master):
    """禁用非主进程的打印，除非在 print() 中指定 force=True。"""
    import builtins as __builtin__
    builtin_print = __builtin__.print
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print

def init_distributed_mode(ddp_sync_file: str):
    """
    初始化 DDP 进程组，使用传入的共享文件进行 FileStore 初始化。
    """
    if 'RANK' not in os.environ:
        print('Not a DDP process. Running in single-process mode.')
        return 0, 1, 0
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    dist_backend = 'gloo'

    init_method = f"file://{ddp_sync_file}"
    print(f"Rank {rank}: Initializing DDP with backend '{dist_backend}' using SHARED FileStore: {init_method}")

    dist.init_process_group(
        backend=dist_backend, 
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    
    torch.cuda.set_device(local_rank)
    print(f"Rank {rank}: DDP Initialized for GPU {local_rank}.")
    return rank, world_size, local_rank

def cleanup():
    """销毁 DDP 进程组。"""
    if dist.is_initialized():
        dist.destroy_process_group()

def setup_logging(log_dir="logs", log_level=logging.INFO, rank=0):
    """
    设置日志。所有进程都在控制台打印，但只有主进程 (rank 0) 会写入文件。
    这个函数现在总是返回一个 logger 对象。
    """
    logger = logging.getLogger(__name__) 
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(log_level)

    console_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if rank == 0:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)

    return logger

def check_gradients(model, epoch, batch_idx):
    """
    打印模型中每一层参数的梯度的统计信息。
    """
    print(f"\n--- Gradient Check @ Epoch {epoch}, Batch {batch_idx} ---")
    total_norm = 0
    
    # DDP 会将模型封装在 .module 中
    model_to_check = model.module if hasattr(model, 'module') else model

    for name, parameter in model_to_check.named_parameters():
        if parameter.grad is not None:
            grad_norm = parameter.grad.norm(2).item()
            grad_mean = parameter.grad.mean().item()
            grad_std = parameter.grad.std().item()
            
            # 格式化打印，让信息更清晰
            print(f"{name:<60} | Grad Norm: {grad_norm:<12.5e} | Grad Mean: {grad_mean:<12.5e} | Grad Std: {grad_std:<12.5e}")
            total_norm += grad_norm ** 2
        else:
            print(f"{name:<60} | Grad Norm: --- None ---") # 打印没有梯度的参数
    
    total_norm = total_norm ** 0.5
    print(f"--- Total Gradient Norm: {total_norm:.5e} ---\n")

def main():
    parser = argparse.ArgumentParser(description="DDP Training Script")
    parser.add_argument("--ddp_sync_file", type=str, required=True,
                        help="Path to the shared file for DDP FileStore initialization.")
    args = parser.parse_args()

    # --- 1. 初始化 DDP 环境 ---
    if args.ddp_sync_file:
        rank, world_size, local_rank = init_distributed_mode(ddp_sync_file=args.ddp_sync_file)
    else: 
        rank, world_size, local_rank = 0, 1, 0
        print("Running in single-process (non-DDP) mode.")

    is_master = (rank == 0)
    setup_for_distributed(is_master)

    logger = setup_logging(rank=rank)
    logger.info("--- Initializing Configurations ---")
    torch.autograd.set_detect_anomaly(True) 
    
    # --- 2. 配置参数 ---
    DEVICE = torch.device(f'cuda:{local_rank}')
    DEFAULT_DTYPE = torch.float32
    torch.set_default_dtype(DEFAULT_DTYPE)

    CONCATENATED_JSON_FILE = "4201359.json"
    RADIAL_CUTOFF = 5.0
    ATOM_FEAT_INIT_CONFIG = {"max_elements": 101, "force_reinit": False}
    MODEL_IRREPS_EDGE_ATTR = "1x0e"
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-6 * world_size
    EPOCHS = 50
    WEIGHT_DECAY = 1e-4
    USE_AMP = True    

    WARMUP_EPOCHS = 2

    ITER_CAPS_IRREPS_STR = "8x0e+4x1o+2x2e" # 保持 Encoder 的输出维度
    ITER_NUM_CAPS = 8

    if is_master:
        print("Pre-initializing atom features to get dimension for model config...")
    _, actual_atom_feat_dim = initialize_atom_features(
        atom_feat_config=ATOM_FEAT_INIT_CONFIG,
        force_reinit=True
    )

    gnn_config = {
        "num_atom_types": ATOM_FEAT_INIT_CONFIG.get("max_elements", 101) + 1,
        "embedding_dim_scalar": 120,
        "irreps_node_hidden": "16x0e+8x1o+4x2e",
        "irreps_node_output": ITER_CAPS_IRREPS_STR,  
        "irreps_edge_attr": "1x0e",
        "irreps_sh": "1x0e+1x1o+1x2e",
        "max_radius": RADIAL_CUTOFF,
        "num_basis_radial": 16,
        "radial_mlp_hidden": [64, 64],
        "num_interaction_layers": 2,
        "use_attention": True,
        "num_attn_heads": 2
    }
    primary_capsule_config = {
        "num_primary_capsules": ITER_NUM_CAPS,
        "irreps_decoded_capsule_feature": ITER_CAPS_IRREPS_STR,
        "capsule_head_activation_hidden_dim": 32,
        "use_separate_decoders": True
    }
    # final_capsule_feature_irreps_example = "4x0e+2x1o"
    # --- Iterative Block 配置 (关键修改) ---
    iterative_block_config = {
        # 推荐增加迭代次数，以发挥残差网络的威力
        "num_main_iterations": 3,    
        "routing_layer_configs": [
            # 每次路由的输入输出胶囊数都相同
            {"num_in_capsules": ITER_NUM_CAPS, "num_out_capsules": ITER_NUM_CAPS, "num_routing_iterations": 3},
            {"num_in_capsules": ITER_NUM_CAPS, "num_out_capsules": ITER_NUM_CAPS, "num_routing_iterations": 3},
            {"num_in_capsules": ITER_NUM_CAPS, "num_out_capsules": ITER_NUM_CAPS, "num_routing_iterations": 3},
        ],
        "gconv_layer_configs": [
            # 每次 gconv 的输出特征维度都相同
            {"irreps_node_output": ITER_CAPS_IRREPS_STR, "irreps_sh": gnn_config["irreps_sh"], "num_basis_radial": gnn_config["num_basis_radial"], "radial_mlp_hidden_dims": [64, 64]},
            {"irreps_node_output": ITER_CAPS_IRREPS_STR, "irreps_sh": gnn_config["irreps_sh"], "num_basis_radial": gnn_config["num_basis_radial"], "radial_mlp_hidden_dims": [64, 64]},
            {"irreps_node_output": ITER_CAPS_IRREPS_STR, "irreps_sh": gnn_config["irreps_sh"], "num_basis_radial": gnn_config["num_basis_radial"], "radial_mlp_hidden_dims": [64, 64]},
        ],
    }

    # --- Final Processor 配置 (修改以匹配统一维度) ---
    final_processor_config = {
        "normalize_aggregation_weights": True,
        # 新增: 明确指定输入的 Irreps
        "input_capsule_feature_irreps_str": ITER_CAPS_IRREPS_STR
    }
    
    # --- Output Layer 配置 (保持不变) ---
    output_layer_config = {
        "cartesian_tensor_formula": "ijk=ikj",
        # 新增: 明确指定输入的 Irreps (从 Final Processor 继承)
        "input_feature_irreps_str": final_processor_config["input_capsule_feature_irreps_str"]
    }
    if is_master:
        print("Configurations initialized.")

    # --- 3. 加载和分发数据集 ---
    full_dataset = None
    if is_master:
        logger.info("\n--- Loading and Processing Dataset (on master process) ---")
        full_dataset = load_piezo_dataset(
            large_json_file_path=CONCATENATED_JSON_FILE,
            structure_key_in_json="structure",        
            piezo_tensor_key_in_json="total", 
            radial_cutoff=RADIAL_CUTOFF,
            device=torch.device('cpu'),
            dtype=DEFAULT_DTYPE,
            atom_feat_config=ATOM_FEAT_INIT_CONFIG,
            irreps_edge_attr_for_data=MODEL_IRREPS_EDGE_ATTR
        )
    
    if world_size > 1:
        dist.barrier()
        object_list = [full_dataset if is_master else None]
        dist.broadcast_object_list(object_list, src=0)
        if not is_master:
            full_dataset = object_list[0]
    
    if full_dataset is None:
        logger.error("Dataset could not be loaded on all processes. Exiting.", force=True)
        if world_size > 1: dist.barrier()
        return

    logger.info(f"Rank {rank}: Total {len(full_dataset)} data points loaded.")

    # --- 4. 划分、归一化和创建分布式 DataLoader ---
    if is_master:
        logger.info("\n--- Splitting, Normalizing, and Augmenting Data ---")
    
    train_dataset, temp_dataset = train_test_split(full_dataset, test_size=0.2, random_state=42, shuffle=True)
    valid_dataset, test_dataset = train_test_split(temp_dataset, test_size=0.5, random_state=42, shuffle=True)
    
    scaler = PiezoTensorScaler().fit(train_dataset)
    model_save_dir = "saved_models_with_norm_and_aug"
    if is_master:
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        scaler.save(os.path.join(model_save_dir, 'piezo_scaler.pth'))

    pos_mean, pos_std = None, None
    if is_master:
        logger.info("Fitting position scaler (mean/std) on the training dataset...")
        all_pos_train = torch.cat([data.pos for data in train_dataset], dim=0)
        pos_mean = all_pos_train.mean(dim=0)
        pos_std = all_pos_train.std(dim=0)
        pos_std[pos_std < 1e-8] = 1.0 # 防止除零
        
        # 保存 scaler 以便将来使用
        torch.save({'mean': pos_mean, 'std': pos_std}, os.path.join(model_save_dir, 'pos_scaler.pth'))
        logger.info("Position scaler fitted and saved.")
    
    # 在 DDP 环境下，将计算好的 mean 和 std 从主进程广播到所有其他进程
    if world_size > 1:
        dist.barrier() # 等待主进程计算和保存完毕
        object_list = [pos_mean, pos_std] if is_master else [None, None]
        dist.broadcast_object_list(object_list, src=0)
        if not is_master:
            pos_mean, pos_std = object_list[0], object_list[1]

    for data in train_dataset: data.y_piezo = scaler.transform(data.y_piezo)
    for data in valid_dataset: data.y_piezo = scaler.transform(data.y_piezo)

    train_transform = Compose([RandomRotatePiezo()])
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, sampler=valid_sampler)

    if is_master:
        logger.info(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)} samples.")
        logger.info(f"Total effective batch size: {BATCH_SIZE * world_size}")

    # --- 5. 模型实例化与 DDP 封装 ---
    if is_master:
        logger.info("\n--- Instantiating Model ---")

    # DEBUG_BYPASS_ENCODER = False  # <--- 激活隔离
    # DEBUG_BYPASS_ROUTING = False # <--- 使用真实路由层
    
    model = EndToEndPiezoNet(
        gnn_config, primary_capsule_config, iterative_block_config, 
        final_processor_config, output_layer_config,
        # 传递两个开关
        # use_dummy_encoder_for_debug=DEBUG_BYPASS_ENCODER,
        # use_dummy_routing_for_debug=DEBUG_BYPASS_ROUTING
    ).to(DEVICE)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False, static_graph=True)
    
    if is_master:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model '{model.module.__class__.__name__}' instantiated with {num_params:,} trainable parameters.")
        logger.info(f"Model moved to {DEVICE} and wrapped with DDP.")

    # --- 6. 定义损失函数和优化器 ---
    if is_master:
        logger.info("\n--- Defining Loss and Optimizer ---")
    criterion = nn.L1Loss(reduction='mean')
    mae_criterion = nn.L1Loss(reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=EPOCHS - WARMUP_EPOCHS, 
        eta_min=1e-6
    )
    scaler_amp = torch.amp.GradScaler(enabled=USE_AMP)

    # --- 7. 训练和验证循环 ---
    if is_master:
        logger.info("\n--- Starting Training ---")
    best_valid_loss = float('inf')

    num_warmup_steps = WARMUP_EPOCHS * len(train_loader)

    for epoch in range(EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        epoch_start_time = time.time()
        
        model.train()
        local_train_loss_sum = 0.0
        local_train_abs_error_sum = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", disable=not is_master)
        for batch_idx, batch_data in enumerate(pbar):

            current_step = epoch * len(train_loader) + batch_idx
            if current_step < num_warmup_steps:
                # 线性增加学习率
                lr_scale = float(current_step + 1) / float(num_warmup_steps)
                for param_group in optimizer.param_groups:
                    # 将学习率设置为 基础学习率 * 缩放比例
                    param_group['lr'] = LEARNING_RATE * lr_scale

            batch_data = train_transform(batch_data).to(DEVICE)
            batch_data.pos = (batch_data.pos - pos_mean.to(DEVICE)) / pos_std.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                predictions_norm = model(batch_data)
                targets_norm = batch_data.y_piezo
                
                # --- Loss 计算 (保持不变) ---
                loss = criterion(predictions_norm.view_as(targets_norm), targets_norm)
                
                # --- MAE 计算 (改为计算总和) ---
                # 我们计算这个 batch 的绝对误差总和，而不是平均值 (MAE)
                abs_error_batch = F.l1_loss(predictions_norm.view_as(targets_norm), targets_norm, reduction='sum')

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss encountered in epoch {epoch+1}. Skipping batch update.")
                continue
        
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)

            if is_master and batch_idx == 0:
                check_gradients(model, epoch + 1, batch_idx + 1)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            # 累加每个 batch 的 loss 和绝对误差总和
            local_train_loss_sum += loss.item() * batch_data.num_graphs
            local_train_abs_error_sum += abs_error_batch.item()

        # 1. 将每个进程的局部总和放入一个 tensor
        train_metrics = torch.tensor([local_train_loss_sum, local_train_abs_error_sum], device=DEVICE)
        
        # 2. 使用 all_reduce 将所有进程的 tensor 求和
        dist.all_reduce(train_metrics, op=dist.ReduceOp.SUM)
        
        # 3. 在主进程上计算全局平均值
        avg_train_loss = float('nan')
        avg_train_mae_norm = float('nan')
        if is_master:
            # train_metrics[0] 是全局的 loss 总和
            # train_metrics[1] 是全局的绝对误差总和
            num_train_samples = len(train_dataset)
            if num_train_samples > 0:
                avg_train_loss = train_metrics[0].item() / num_train_samples
                
                # MAE 是总绝对误差 / (样本数 * 每个样本的元素数)
                num_elements_per_sample = train_dataset[0].y_piezo.numel()
                avg_train_mae_norm = train_metrics[1].item() / (num_train_samples * num_elements_per_sample)

        # --- Validation ---
        model.eval()
        valid_loss_sum = 0.0
        # --- 新增：用于计算 MAE 的误差总和 ---
        valid_abs_error_sum = 0.0
        
        if valid_loader:
            with torch.no_grad():
                for batch_data_val in valid_loader:
                    batch_data_val = batch_data_val.to(DEVICE)
                    batch_data_val.pos = (batch_data_val.pos - pos_mean.to(DEVICE)) / pos_std.to(DEVICE)
                    
                    predictions_norm_val = model(batch_data_val)
                    targets_norm_val = batch_data_val.y_piezo
                    
                    # 计算 Loss
                    loss_val = criterion(predictions_norm_val.view_as(targets_norm_val), targets_norm_val)
                    if not torch.isnan(loss_val):
                        valid_loss_sum += loss_val.item() * batch_data_val.num_graphs

                    # --- 新的 MAE 计算方式 ---
                    # 1. 将预测和目标逆变换回原始尺度
                    preds_orig = scaler.inverse_transform(predictions_norm_val)
                    targets_orig = scaler.inverse_transform(targets_norm_val)
                    
                    # 2. 计算这个 batch 的绝对误差总和 (而不是 MAE)
                    # L1Loss(reduction='sum')
                    abs_error_sum_batch = F.l1_loss(preds_orig, targets_orig, reduction='sum')
                    valid_abs_error_sum += abs_error_sum_batch.item()
                    # 我们不再需要 all_preds_orig_list 和 all_targets_orig_list

        # --- 新的 DDP 聚合逻辑 ---
        # 1. 将每个进程计算出的 loss_sum 和 abs_error_sum 放入一个 tensor 中
        valid_metrics = torch.tensor([valid_loss_sum, valid_abs_error_sum], device=DEVICE)
        
        # 2. 使用 all_reduce 将所有进程的 metrics tensor 求和
        dist.all_reduce(valid_metrics, op=dist.ReduceOp.SUM)
        
        # 3. 在主进程上计算最终的平均值
        avg_valid_loss = float('inf')
        avg_valid_mae = float('inf')
        if is_master:
            # valid_metrics[0] 现在是全局的 loss 总和
            # valid_metrics[1] 现在是全局的绝对误差总和
            
            # 使用验证集的总长度来计算平均值
            num_valid_samples = len(valid_dataset)
            if num_valid_samples > 0:
                avg_valid_loss = valid_metrics[0].item() / num_valid_samples
                
                # MAE 是总绝对误差 / (样本数 * 每个样本的元素数)
                # 我们的目标 y 是 (3,3,3) 张量，有 27 个元素
                num_elements_per_sample = valid_dataset[0].y_piezo.numel()
                avg_valid_mae = valid_metrics[1].item() / (num_valid_samples * num_elements_per_sample)

        # --- 日志记录与模型保存 (只在主进程执行) ---
        if is_master:
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            if epoch >= WARMUP_EPOCHS:
                scheduler.step()
            
            log_message = (
                f"Epoch {epoch+1:03d}/{EPOCHS:03d} | "
                f"Train Loss: {avg_train_loss:.4e} | "
                f"Train MAE (Norm): {avg_train_mae_norm:.4e} | " 
                f"Valid Loss: {avg_valid_loss:.4e} | "
                f"Valid MAE (Orig): {avg_valid_mae:.4e} | " 
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {epoch_duration:.1f}s"
            )
            logger.info(log_message)
            
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                model_save_path = os.path.join(model_save_dir, "model_best.pth")
                torch.save({'model_state_dict': model.module.state_dict()}, model_save_path)
                logger.info(f"Model improved and saved to {model_save_path}")

    if is_master:
        logger.info("--- Training Finished ---")
    
    cleanup()

if __name__ == '__main__':
    main()