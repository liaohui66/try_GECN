# geometry_utils.py

import torch
import torch.nn.functional as F
import math
import os

if not os.path.exists("debug_tensors"):
    os.makedirs("debug_tensors")
problematic_tensor_count = 0

def hat(v: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hat operator of a batch of 3D vectors.
    """
    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")
    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)
    x, y, z = v.unbind(1)
    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x
    return h

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Computes torch.sqrt(torch.max(0, x))
    """
    return torch.sqrt(torch.clamp(x, min=0.0))

def standardize_quaternion(quaternions: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert a batch of quaternions to standard form.
    """
    # Ensure real part is non-negative
    real_part_sign = torch.sign(quaternions[..., :1])
    # Handle the case where real part is exactly zero
    real_part_sign = torch.where(real_part_sign == 0, torch.ones_like(real_part_sign), real_part_sign)
    q_std = quaternions * real_part_sign.expand_as(quaternions)
    # Normalize to unit quaternion
    return q_std / torch.linalg.norm(q_std, dim=-1, keepdim=True).clamp(min=eps)

def axis_angle_to_quaternion(axis_angle: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert an axis-angle representation to a quaternion.
    Expects axis_angle to be a tensor where the last dimension is 3.
    The norm of the axis_angle vector is the angle of rotation,
    and the normalized vector is the axis of rotation.
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5

    small_angle_threshold = 1e-4 
    small_angles_mask = angles < small_angle_threshold
    
    cos_half_angles = torch.cos(half_angles)
        
    sin_half_div_angle = torch.zeros_like(angles) 
    
    large_mask = ~small_angles_mask
    if large_mask.any():
        denom_large = angles[large_mask].clamp(min=eps)
        sin_half_div_angle[large_mask] = torch.sin(half_angles[large_mask]) / denom_large

    if small_angles_mask.any():
        theta_sq_small = angles[small_angles_mask] ** 2
        sin_half_div_angle[small_angles_mask] = 0.5 - theta_sq_small / 48.0

    vector_part = axis_angle * sin_half_div_angle
    
    quaternions = torch.cat(
        [cos_half_angles, vector_part], dim=-1
    )

    return quaternions

def quaternion_to_axis_angle(quaternions: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert a quaternion to an axis-angle representation.
    Input quaternion is expected to be in [w, x, y, z] format.
    The output axis_angle is a 3D vector where the direction is the axis
    and the norm is the angle of rotation (typically in [0, pi] or [0, 2pi]
    depending on the convention used for standardizing w).
    This implementation aims for angles in [0, 2pi] initially, based on acos,
    and standardizes w to be non-negative.
    If w becomes ~1 (angle is ~0) or ~-1 (angle is ~2pi), special handling for axis.
    """
    q_norm = quaternions / torch.linalg.norm(quaternions, dim=-1, keepdim=True).clamp(min=eps)
    
    w_negative_mask = q_norm[..., 0] < 0.0
    flip_sign = torch.ones_like(q_norm[..., :1])
    
    mask_for_flip = w_negative_mask
    if w_negative_mask.ndim < flip_sign.ndim:
        mask_for_flip = w_negative_mask.unsqueeze(-1)
    flip_sign[mask_for_flip.expand_as(flip_sign)] = -1.0

    q_standardized = q_norm * flip_sign.expand_as(q_norm)
    
    w = q_standardized[..., 0:1]
    xyz = q_standardized[..., 1:]

    acos_clamp_eps = 1e-6 
    w_clamped = torch.clamp(w, -1.0 + acos_clamp_eps, 1.0 - acos_clamp_eps)
    half_angles = torch.acos(w_clamped)
    angles = 2.0 * half_angles

    small_angle_threshold = 1e-4
    small_angles_mask = angles < small_angle_threshold

    axis_angle = torch.zeros_like(xyz)

    idx_small_angles = small_angles_mask.squeeze(-1) # Shape (...,) for indexing
    if idx_small_angles.any():
        # angles[small_angles_mask] will be 1D (k_small,) if angles was (N,1)
        # We need to keep the last dim for scale_factor to be (k_small, 1)
        theta_small_for_taylor = angles[small_angles_mask].unsqueeze(-1) # Ensure shape (k_small, 1)
        # Or more robustly, ensure angles itself is used with the mask correctly
        # theta_small_for_taylor = angles[small_angles_mask.expand_as(angles)].reshape(-1,1) # Gets elements and reshapes
        # A simpler way given angles is (N,1) and small_angles_mask is (N,1):
        # theta_small_for_taylor = angles[small_angles_mask] # This will select elements and make it 1D (k_small,)
        # Let's ensure it stays (k_small, 1)
        
        # Correct selection if angles is (N,1) and small_angles_mask is (N,1)
        # This will get elements from angles where mask is true, result is (k_small, 1)
        theta_small_for_taylor_masked = angles[small_angles_mask.squeeze(-1)] # If angles (N,1) and mask (N,1) -> squeeze mask to (N,)
                                                                            # then result is (k_small,1)

        theta_sq_small_for_taylor = theta_small_for_taylor_masked ** 2 
        scale_factor_small = 2.0 + theta_sq_small_for_taylor / 12.0 + (theta_sq_small_for_taylor**2) / 480.0
        # scale_factor_small is now (k_small, 1)
        
        xyz_small_branch = xyz[idx_small_angles] # Shape (k_small, 3)
        axis_angle_small_calc = xyz_small_branch * scale_factor_small # (k_small, 3) * (k_small, 1) -> (k_small, 3)
        axis_angle[idx_small_angles] = axis_angle_small_calc

    # --- Large Angle Branch ---
    idx_large_angles = (~small_angles_mask).squeeze(-1) # Shape (...,)
    if idx_large_angles.any():
        # Similar to above, ensure (k_large, 1) shape for angle-derived terms
        angles_large_branch_masked = angles[(~small_angles_mask).squeeze(-1)] # (k_large, 1)
        half_angles_large_branch_masked = half_angles[(~small_angles_mask).squeeze(-1)] # (k_large, 1)
        
        xyz_large_branch = xyz[idx_large_angles] # Shape (k_large, 3)

        sin_half_angles_large = torch.sin(half_angles_large_branch_masked) # (k_large, 1)
        sin_half_angles_safe = sin_half_angles_large.clamp(min=eps) # (k_large, 1)
        
        scale_factor_large = angles_large_branch_masked / sin_half_angles_safe # (k_large, 1)
        
        axis_angle_large_calc = xyz_large_branch * scale_factor_large # (k_large, 3) * (k_large, 1) -> (k_large, 3)
        axis_angle[idx_large_angles] = axis_angle_large_calc
        
    return axis_angle

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of quaternions to rotation matrices.
    Input quaternions are expected in [w, x, y, z] format (scalar-first).
    The function normalizes the input quaternions before conversion.
    """
    q_norm = quaternions / torch.linalg.norm(quaternions, dim=-1, keepdim=True).clamp(min=1e-8)

    r, i, j, k = torch.unbind(q_norm, dim=-1)

    two_s = 2.0 

    o = torch.stack(
        [
            1 - two_s * (j * j + k * k),    two_s * (i * j - k * r),    two_s * (i * k + j * r),
            two_s * (i * j + k * r),    1 - two_s * (i * i + k * k),    two_s * (j * k - i * r),
            two_s * (i * k - j * r),    two_s * (j * k + i * r),    1 - two_s * (i * i + j * j),
        ],
        dim=-1,
    )
    matrix = o.reshape(quaternions.shape[:-1] + (3, 3))
        
    return matrix

def matrix_to_quaternion(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert a batch of rotation matrices to quaternions [w, x, y, z] (scalar-first).
    Uses Shepperd's method for numerical stability.
    The output quaternion is standardized (w >= 0 and unit norm).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Input matrix must be a batch of 3x3 matrices. Got shape {matrix.shape}.")

    batch_dims = matrix.shape[:-2]

    matrix_flat = matrix.reshape(-1, 3, 3)
    num_batch_elements = matrix_flat.shape[0]

    m00, m01, m02 = matrix_flat[:, 0, 0], matrix_flat[:, 0, 1], matrix_flat[:, 0, 2]
    m10, m11, m12 = matrix_flat[:, 1, 0], matrix_flat[:, 1, 1], matrix_flat[:, 1, 2]
    m20, m21, m22 = matrix_flat[:, 2, 0], matrix_flat[:, 2, 1], matrix_flat[:, 2, 2]

    q_abs_sq_times_4 = torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], dim=-1)

    # q_abs_sq_times_4 = torch.clamp(q_abs_sq_times_4, min=0.0)
    # eps = 1e-8 
    # q_abs = torch.sqrt(q_abs_sq_times_4 + eps)
    q_abs_sq_f32 = q_abs_sq_times_4.to(torch.float32)
    
    # 在 float32 下进行安全的开方
    # 我们使用一个稍大一点的 epsilon 来增加鲁棒性
    stable_eps = 1e-6 
    q_abs_f32 = torch.sqrt(torch.clamp(q_abs_sq_f32, min=stable_eps))
    
    # 将结果转换回原始数据类型，以便后续计算
    q_abs = q_abs_f32.to(matrix.dtype)

    max_idx = torch.argmax(q_abs, dim=-1) # Shape (B,)

    quat = torch.zeros(num_batch_elements, 4, device=matrix.device, dtype=matrix.dtype)

    w_largest_mask = (max_idx == 0)
    if w_largest_mask.any():
        q0_abs_val = q_abs[w_largest_mask, 0] 
        quat[w_largest_mask, 0] = q0_abs_val * 0.5

        denom_w = (2.0 * q0_abs_val).clamp(min=eps)
        quat[w_largest_mask, 1] = (m21[w_largest_mask] - m12[w_largest_mask]) / denom_w
        quat[w_largest_mask, 2] = (m02[w_largest_mask] - m20[w_largest_mask]) / denom_w
        quat[w_largest_mask, 3] = (m10[w_largest_mask] - m01[w_largest_mask]) / denom_w

    x_largest_mask = (max_idx == 1)
    if x_largest_mask.any():
        q1_abs_val = q_abs[x_largest_mask, 1] 
        quat[x_largest_mask, 1] = q1_abs_val * 0.5 
        denom_x = (2.0 * q1_abs_val).clamp(min=eps)
        quat[x_largest_mask, 0] = (m21[x_largest_mask] - m12[x_largest_mask]) / denom_x
        quat[x_largest_mask, 2] = (m10[x_largest_mask] + m01[x_largest_mask]) / denom_x
        quat[x_largest_mask, 3] = (m02[x_largest_mask] + m20[x_largest_mask]) / denom_x

    y_largest_mask = (max_idx == 2)
    if y_largest_mask.any():
        q2_abs_val = q_abs[y_largest_mask, 2] 
        quat[y_largest_mask, 2] = q2_abs_val * 0.5 
        denom_y = (2.0 * q2_abs_val).clamp(min=eps)
        quat[y_largest_mask, 0] = (m02[y_largest_mask] - m20[y_largest_mask]) / denom_y
        quat[y_largest_mask, 1] = (m10[y_largest_mask] + m01[y_largest_mask]) / denom_y
        quat[y_largest_mask, 3] = (m21[y_largest_mask] + m12[y_largest_mask]) / denom_y

    z_largest_mask = (max_idx == 3)
    if z_largest_mask.any():
        q3_abs_val = q_abs[z_largest_mask, 3] 
        quat[z_largest_mask, 3] = q3_abs_val * 0.5 
        denom_z = (2.0 * q3_abs_val).clamp(min=eps)
        quat[z_largest_mask, 0] = (m10[z_largest_mask] - m01[z_largest_mask]) / denom_z
        quat[z_largest_mask, 1] = (m20[z_largest_mask] + m02[z_largest_mask]) / denom_z
        quat[z_largest_mask, 2] = (m12[z_largest_mask] + m21[z_largest_mask]) / denom_z

    final_quat = standardize_quaternion(quat.reshape(batch_dims + (4,)), eps=eps)
        
    return final_quat

def so3_exp_map_custom_via_quat(log_rot: torch.Tensor) -> torch.Tensor:
    """
    SO(3) exponential map: axis_angle -> quaternion -> rotation matrix.
    Args:
        log_rot: Tensor of axis-angle vectors, shape (..., 3).
    Returns:
        Tensor of rotation matrices, shape (..., 3, 3).
    """
    quaternions = axis_angle_to_quaternion(log_rot)
    return quaternion_to_matrix(quaternions)

def so3_log_map_custom_via_quat(R: torch.Tensor) -> torch.Tensor:
    """
    SO(3) logarithmic map: rotation matrix -> quaternion -> axis_angle.
    Args:
        R: Tensor of rotation matrices, shape (..., 3, 3).
    Returns:
        Tensor of axis-angle vectors, shape (..., 3).
    """
    quaternions = matrix_to_quaternion(R)
    return quaternion_to_axis_angle(quaternions)

def _so3_exp_map_for_se3(log_rot: torch.Tensor, eps: float = 1e-6):
    """
    Internal helper to compute components needed for SE(3) from an SO(3) log_rot.
    Calculates R, rotation angle, skew-symmetric matrix of log_rot, and its square.
    Args:
        log_rot: Axis-angle vectors, shape (..., 3).
        eps: Small epsilon for clamping in sqrt to ensure numerical stability.
    Returns:
        R: Rotation matrices, shape (..., 3, 3).
        rot_angles: Rotation angles (norm of log_rot), shape (...,).
        skews: Skew-symmetric matrices of log_rot, shape (..., 3, 3).
        skews_square: Square of skews, shape (..., 3, 3).
    """
    if log_rot.shape[-1] != 3:
        raise ValueError("Input log_rot must have the last dimension of size 3.")

    rot_angles_sq = torch.sum(log_rot * log_rot, dim=-1)
    rot_angles = torch.sqrt(rot_angles_sq.clamp(min=eps))

    skews = hat(log_rot)
    skews_square = torch.bmm(skews.reshape(-1, 3, 3), skews.reshape(-1, 3, 3))

    original_batch_dims = log_rot.shape[:-1]
    skews_square = skews_square.reshape(original_batch_dims + (3,3) if len(original_batch_dims) > 0 else (3,3) )

    quaternions = axis_angle_to_quaternion(log_rot) 
    R = quaternion_to_matrix(quaternions)
            
    return R, rot_angles, skews, skews_square

def _get_se3_V_input_custom(log_rotation: torch.Tensor, eps: float = 1e-4):
    """
    Helper to get inputs for SE(3) V matrix computation from log_rotation.
    Args:
        log_rotation: Axis-angle vectors for the rotational part, shape (..., 3).
        eps: Epsilon for numerical stability in the underlying _so3_exp_map_for_se3 call
             (specifically for its sqrt().clamp(min=eps)).
    Returns:
        log_rotation: The input log_rotation (passed through).
        log_rotation_hat: Skew-symmetric matrix of log_rotation, shape (..., 3, 3).
        log_rotation_hat_square: Square of log_rotation_hat, shape (..., 3, 3).
        rotation_angles: Norm of log_rotation, shape (...,).
    """
    _, rotation_angles, log_rotation_hat, log_rotation_hat_square = _so3_exp_map_for_se3(
        log_rotation, 
        eps=eps
    )
    return log_rotation, log_rotation_hat, log_rotation_hat_square, rotation_angles

def _se3_V_matrix_custom(
    log_rotation_ref_for_eye: torch.Tensor, 
    log_rotation_hat: torch.Tensor,         
    log_rotation_hat_square: torch.Tensor,  
    rotation_angles: torch.Tensor,          
    eps: float = 1e-4                       
) -> torch.Tensor:
    """
    Computes the V matrix for SE(3) exponential map.
    V = I + ((1-cos(theta))/theta^2) * K + ((theta-sin(theta))/theta^3) * K^2
    Args:
        log_rotation_ref_for_eye: A reference tensor (e.g., original log_rotation) 
                                  to get dtype and device for torch.eye.
        log_rotation_hat: Skew-symmetric matrix K, shape (..., 3, 3).
        log_rotation_hat_square: K^2, shape (..., 3, 3).
        rotation_angles: Rotation angles theta, shape (...,).
        eps: Epsilon for determining when to use Taylor expansion for coefficients.
    Returns:
        V matrix, shape (..., 3, 3).
    """
    theta_sq = rotation_angles**2

    small_angles_mask_v = rotation_angles < eps 

    coef_K = torch.empty_like(rotation_angles)
    coef_K[small_angles_mask_v] = 0.5 - theta_sq[small_angles_mask_v] / 24.0

    safe_theta_sq_denom_K = theta_sq[~small_angles_mask_v].clamp(min=1e-12)
    coef_K[~small_angles_mask_v] = (1 - torch.cos(rotation_angles[~small_angles_mask_v])) / safe_theta_sq_denom_K

    coef_K_sq = torch.empty_like(rotation_angles)
    coef_K_sq[small_angles_mask_v] = (1.0/6.0) - theta_sq[small_angles_mask_v] / 120.0

    safe_theta_cub_denom_K_sq = (rotation_angles[~small_angles_mask_v]**3).clamp(min=1e-12)
    coef_K_sq[~small_angles_mask_v] = (rotation_angles[~small_angles_mask_v] - torch.sin(rotation_angles[~small_angles_mask_v])) / safe_theta_cub_denom_K_sq

    V = (
        torch.eye(3, dtype=log_rotation_ref_for_eye.dtype, device=log_rotation_ref_for_eye.device)[None]
        + coef_K.unsqueeze(-1).unsqueeze(-1) * log_rotation_hat
        + coef_K_sq.unsqueeze(-1).unsqueeze(-1) * log_rotation_hat_square
    )
    return V

def se3_exp_map_custom(log_transform: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    SE(3) exponential map: converts a 6D twist vector (se(3) algebra) to a 4x4 SE(3) matrix.
    Args:
        log_transform: Tensor of 6D twists [translation_part, rotation_part_axis_angle],
                       shape (N, 6).
        eps: Epsilon value used for numerical stability in intermediate calculations,
             primarily for V-matrix coefficients and sqrt clamps in SO(3) part.
    Returns:
        Tensor of 4x4 SE(3) transformation matrices, shape (N, 4, 4).
    """
    if not (log_transform.ndim == 2 and log_transform.shape[1] == 6):
        raise ValueError(f"Expected input log_transform to be of shape (N, 6). Got {log_transform.shape}")
    
    N = log_transform.shape[0]
    log_translation = log_transform[..., :3]  
    log_rotation = log_transform[..., 3:]    

    R, rotation_angles, log_rotation_hat, log_rotation_hat_square = _so3_exp_map_for_se3(
        log_rotation, 
        eps=eps 
    )

    V = _se3_V_matrix_custom(
        log_rotation, 
        log_rotation_hat, 
        log_rotation_hat_square, 
        rotation_angles, 
        eps=eps 
    )

    T_matrix_translation_part = torch.bmm(V, log_translation.unsqueeze(-1)).squeeze(-1)

    transform_matrix = torch.zeros(N, 4, 4, dtype=log_transform.dtype, device=log_transform.device)
    transform_matrix[:, :3, :3] = R
    transform_matrix[:, :3, 3] = T_matrix_translation_part
    transform_matrix[:, 3, 3] = 1.0

    if torch.isnan(transform_matrix).any() or torch.isinf(transform_matrix).any():
        global problematic_tensor_count
        problematic_tensor_count += 1
        
        print("\n" + "#"*70)
        print(f"!!! EVIDENCE FOUND (Instance #{problematic_tensor_count}): Instability in se3_exp_map_custom !!!")
        print("    - The output transformation matrix contains NaN/Inf.")
        
        # 找出是哪个输入 log_transform 导致了这个问题
        problematic_mask = torch.isnan(transform_matrix).any(dim=(1,2)) | torch.isinf(transform_matrix).any(dim=(1,2))
        problematic_input = log_transform[problematic_mask]
        
        print(f"    - Number of problematic input vectors: {problematic_input.shape[0]}")
        print(f"    - Max absolute value in problematic inputs: {torch.max(torch.abs(problematic_input)).item()}")
        print("    - This strongly suggests the input 'log_transform' has excessively large values.")
        
        # 保存罪魁祸首
        save_path = f"debug_tensors/problematic_log_transform_input_{problematic_tensor_count}.pt"
        torch.save(log_transform.detach().cpu(), save_path)
        print(f"    - Saved the FULL input tensor to: {save_path}")
        print("#"*70 + "\n")

        # 为了防止程序崩溃，用一个安全的单位矩阵替换掉坏数据
        identity_matrix = torch.eye(4, dtype=transform_matrix.dtype, device=transform_matrix.device)
        # 只替换那些有问题的位置
        transform_matrix = torch.where(problematic_mask.view(-1, 1, 1), identity_matrix, transform_matrix)
    
    return transform_matrix

grad_check_results_geo = {}
def get_grad_hook_geo(name):
    def hook(grad):
        if grad is not None:
            has_nan = torch.isnan(grad).any()
            has_inf = torch.isinf(grad).any()
            if has_nan or has_inf:
                grad_check_results_geo[name] = {'has_nan': has_nan.item(), 'has_inf': has_inf.item()}
    return hook

def se3_log_map_custom(transform: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    SE(3) logarithmic map, MODIFIED WITH DIAGNOSTIC PROBES.
    """
    if not (transform.ndim == 3 and transform.shape[1:] == (4, 4)):
        raise ValueError(f"Input transform must be of shape (N, 4, 4). Got {transform.shape}")

    R = transform[:, :3, :3] 
    T = transform[:, :3, 3] 

    log_rotation = so3_log_map_custom_via_quat(R) 

    _, rot_angles_for_V, skews_for_V, skews_square_for_V = _so3_exp_map_for_se3(
        log_rotation, 
        eps=eps 
    )

    V = _se3_V_matrix_custom(
        log_rotation,         
        skews_for_V,          
        skews_square_for_V,   
        rot_angles_for_V,     
        eps=eps               
    )

    # 1. 高精度计算作为安全参考
    device_type = V.device.type
    with torch.amp.autocast(device_type=device_type, enabled=False):
        V_f32, T_f32 = V.to(torch.float32), T.to(torch.float32)
        try:
            log_translation_ref_f32 = torch.linalg.solve(V_f32, T_f32.unsqueeze(-1)).squeeze(-1)
            ref_is_invalid = torch.isnan(log_translation_ref_f32).any() or torch.isinf(log_translation_ref_f32).any()
        except torch.linalg.LinAlgError:
            ref_is_invalid = True

    try:
        T_matched_dtype = T.to(V.dtype)
        
        log_translation_original = torch.linalg.solve(V, T_matched_dtype.unsqueeze(-1)).squeeze(-1)
        original_is_nan = torch.isnan(log_translation_original).any() or torch.isinf(log_translation_original).any()
    except torch.linalg.LinAlgError:
        original_is_nan = True

    # 3. 对比分析
    evidence_found = False
    final_log_translation = None

    if original_is_nan and not ref_is_invalid:
        evidence_found, reason = True, "Low-precision result is NaN/Inf, high-precision is clean."
    elif not original_is_nan and not ref_is_invalid:
        relative_diff = torch.norm(log_translation_original.float() - log_translation_ref_f32) / (torch.norm(log_translation_ref_f32) + 1e-6)
        if relative_diff > 1.0: # 设定一个巨大的相对差异阈值
            evidence_found, reason = True, f"Huge relative difference ({relative_diff:.2f}) between precisions."
            
    if evidence_found:
        problematic_tensor_count += 1
        print("\n" + "!"*70)
        print(f"!!! EVIDENCE FOUND (Instance #{problematic_tensor_count}): Instability in torch.linalg.solve !!!")
        print(f"    - REASON: {reason}")
        save_path_v = f"debug_tensors/problematic_V_{problematic_tensor_count}.pt"
        torch.save(V.detach().cpu(), save_path_v)
        print(f"    - Saved problematic V matrix to: {save_path_v}")
        print("!"*70 + "\n")
        
        # 发现问题时，使用更安全的高精度结果继续运行
        final_log_translation = log_translation_ref_f32.to(transform.dtype)
    else:
        # 未发现问题，或高精度也失败，则使用原始结果（或回退值）
        final_log_translation = torch.zeros_like(T) if original_is_nan else log_translation_original

    
    log_transform_out = torch.cat((final_log_translation, log_rotation), dim=1)
    return log_transform_out

def se3_mean_algebra_custom(
    log_se3_poses: torch.Tensor,
    weights: torch.Tensor,
    normalize_weights_method: str = "softmax",
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute weighted mean of SE(3) poses in algebra space.
    """
    if log_se3_poses.shape[-1] != 6:
        raise ValueError("Last dimension of log_se3_poses must be 6.")
    
    if weights.ndim == log_se3_poses.ndim - 1:
        weights = weights.unsqueeze(-1)
    
    if weights.shape[:-1] != log_se3_poses.shape[:-1]:
        raise ValueError(f"Shape mismatch between log_se3_poses {log_se3_poses.shape} and weights {weights.shape}")

    # Normalize weights
    if normalize_weights_method == "softmax":
        normalized_weights = F.softmax(weights, dim=-2)
    elif normalize_weights_method == "sum":
        normalized_weights = weights / weights.sum(dim=-2, keepdim=True).clamp(min=eps)
    elif normalize_weights_method == "none":
        normalized_weights = weights
    else:
        raise ValueError(f"Unknown normalize_weights_method: {normalize_weights_method}")

    # Split translations and rotations
    log_translations = log_se3_poses[..., :3]
    log_rotations = log_se3_poses[..., 3:]

    # Compute mean translation
    mean_log_translation = (log_translations * normalized_weights).sum(dim=-2)

    # Prepare for quaternion averaging
    original_shape_prefix = log_rotations.shape[:-2]
    num_poses_to_avg = log_rotations.shape[-2]

    flat_log_rotations = log_rotations.reshape(-1, 3)
    flat_quaternions = axis_angle_to_quaternion(flat_log_rotations)

    quaternions_for_avg = flat_quaternions.reshape(original_shape_prefix + (num_poses_to_avg, 4))
    weights_for_avg = normalized_weights.reshape(original_shape_prefix + (num_poses_to_avg, 1))

    # Compute weighted average quaternion
    avg_quat_unnorm = (quaternions_for_avg * weights_for_avg).sum(dim=-2)
    avg_quat_normalized = avg_quat_unnorm / torch.linalg.norm(avg_quat_unnorm, dim=-1, keepdim=True).clamp(min=eps)

    # Convert quaternion back to axis-angle
    mean_log_rotation = quaternion_to_axis_angle(avg_quat_normalized)

    # Combine translation and rotation
    mean_se3_algebra = torch.cat([mean_log_translation, mean_log_rotation], dim=-1)
    return mean_se3_algebra


def se3_inverse_custom(T_mat: torch.Tensor) -> torch.Tensor:
    """
    Computes the inverse of a batch of SE(3) 4x4 transformation matrices.
    If T = [R | t]
           [0 | 1],
    then T_inv = [R^T | -R^T @ t]
                 [0   |     1    ].
    Args:
        T_mat: Tensor of SE(3) matrices, shape (..., 4, 4).
    Returns:
        Tensor of inverse SE(3) matrices, shape (..., 4, 4).
    """
    if T_mat.shape[-2:] != (4, 4):
        raise ValueError(f"Input T_mat must be a batch of 4x4 matrices. Got shape {T_mat.shape}.")

    R = T_mat[..., :3, :3]
    t = T_mat[..., :3, 3]

    R_T = R.transpose(-2, -1) 

    t_inv = -torch.matmul(R_T, t.unsqueeze(-1)).squeeze(-1)

    T_inv_mat_out = torch.zeros_like(T_mat) 
    T_inv_mat_out[..., :3, :3] = R_T
    T_inv_mat_out[..., :3, 3] = t_inv
    T_inv_mat_out[..., 3, :3] = 0.0 
    T_inv_mat_out[..., 3, 3] = 1.0
    
    return T_inv_mat_out

def se3_distance_algebra_custom(
    log_se3_pose1: torch.Tensor, 
    log_se3_pose2: torch.Tensor,
    rot_weight: float = 1.0,
    eps_sqrt_clamp: float = 1e-8
) -> torch.Tensor:
    """
    Computes a weighted distance between two SE(3) poses given in their
    6D algebra (twist) representation.
    Distance = sqrt( ||log(T1^-1 * T2)||_trans^2 + rot_weight * ||log(T1^-1 * T2)||_rot^2 )
    Args:
        log_se3_pose1: Twist vector for the first pose, shape (..., 6).
        log_se3_pose2: Twist vector for the second pose, shape (..., 6).
        rot_weight: Weight applied to the rotational component of the distance.
        eps_sqrt_clamp: Small epsilon for clamping the argument of the final sqrt
                        to prevent issues with tiny negative values due to precision.
    Returns:
        Computed distance, shape (...).
    """
    if not (log_se3_pose1.shape == log_se3_pose2.shape and log_se3_pose1.shape[-1] == 6):
        raise ValueError("Inputs log_se3_pose1 and log_se3_pose2 must have the same shape ending in 6.")

    T1_mat = se3_exp_map_custom(log_se3_pose1)
    T2_mat = se3_exp_map_custom(log_se3_pose2)

    T1_inv_mat = se3_inverse_custom(T1_mat) 
    T_rel_mat = torch.matmul(T1_inv_mat, T2_mat)

    log_T_rel_algebra = se3_log_map_custom(T_rel_mat)

    log_trans_rel = log_T_rel_algebra[..., :3]
    log_rot_rel = log_T_rel_algebra[..., 3:]

    dist_trans_sq = torch.sum(log_trans_rel ** 2, dim=-1)
    dist_rot_sq = torch.sum(log_rot_rel ** 2, dim=-1)

    weighted_dist_sq = dist_trans_sq + rot_weight * dist_rot_sq

    distance = torch.sqrt(weighted_dist_sq.clamp(min=eps_sqrt_clamp))

    return distance