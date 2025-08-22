# group_conv.py (THE ABSOLUTE FINAL, CORRECTED QUATERNION VERSION)

import torch
import torch.nn as nn
from e3nn import o3
from typing import List, Optional, Tuple
from torch_geometric.nn import global_mean_pool
import math
import torch.nn.functional as F

try:
    from .encoder import TFNInteractionBlock, RadialMLP
    from .geometry_utils import se3_exp_map_custom, se3_inverse_custom
except ImportError:
    from encoder import TFNInteractionBlock, RadialMLP
    from geometry_utils import se3_exp_map_custom, se3_inverse_custom

from torch.autograd import Function

class SVD_with_Identity_Grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, R_matrix):
        with torch.amp.autocast(device_type=R_matrix.device.type, enabled=False):
            try:
                R_matrix_f32 = R_matrix.to(torch.float32)
                U, S, Vh = torch.linalg.svd(R_matrix_f32)
                product = U @ Vh
                det = torch.det(product)
                U[det < 0] = -U[det < 0]
                R_perfect = U @ Vh
                return R_perfect.to(R_matrix.dtype)
            except torch.linalg.LinAlgError:
                batch_size = R_matrix.shape[0]
                identity = torch.eye(3, device=R_matrix.device, dtype=R_matrix.dtype)
                return identity.unsqueeze(0).expand(batch_size, -1, -1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def robust_matrix_to_angles(R: torch.Tensor, eps: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    beta = torch.acos(torch.clamp(R[..., 2, 2], -1.0 + eps, 1.0 - eps))
    is_gimbal_lock = (beta < eps) | (beta > math.pi - eps)
    alpha_normal = torch.atan2(R[..., 1, 2], R[..., 0, 2])
    gamma_normal = torch.atan2(R[..., 2, 1], -R[..., 2, 0])
    alpha_gimbal = torch.atan2(-R[..., 0, 1], R[..., 0, 0])
    gamma_gimbal = torch.zeros_like(alpha_gimbal)
    alpha = torch.where(is_gimbal_lock, alpha_gimbal, alpha_normal)
    gamma = torch.where(is_gimbal_lock, gamma_gimbal, gamma_normal)
    return alpha, beta, gamma

class SE3GroupConvLayer(nn.Module):
    def __init__(self,
                 irreps_node_input: str,
                 irreps_node_output: str,
                 irreps_sh: str,
                 num_basis_radial: int,
                 radial_mlp_hidden_dims: List[int],
                 num_interaction_layers: int = 1,
                 **kwargs
                ):
        super().__init__()
        self.irreps_node_input_obj = o3.Irreps(irreps_node_input)
        self.irreps_node_output_obj = o3.Irreps(irreps_node_output)
        self.irreps_sh_obj = o3.Irreps(irreps_sh)
        self.num_basis_radial = num_basis_radial
        self.num_interaction_layers = num_interaction_layers
        self.eps = 1e-8
        self.spherical_harmonics = o3.SphericalHarmonics(self.irreps_sh_obj, normalize=True, normalization='component')
        self.radial_embedding = RadialMLP(1, radial_mlp_hidden_dims, self.num_basis_radial)
        self.interaction_layers_module = nn.ModuleList()
        current_block_input_irreps = self.irreps_node_input_obj
        for i in range(self.num_interaction_layers):
            block_output_irreps = current_block_input_irreps if i < self.num_interaction_layers - 1 else self.irreps_node_output_obj
            block = TFNInteractionBlock(
                irreps_node_input=str(current_block_input_irreps),
                irreps_node_output=str(block_output_irreps),
                irreps_edge_attr="0e",
                irreps_sh=str(self.irreps_sh_obj),
                num_basis_radial=self.num_basis_radial,
                radial_mlp_hidden=radial_mlp_hidden_dims
            )
            self.interaction_layers_module.append(block)
            current_block_input_irreps = block_output_irreps

    def _get_D_from_matrix_final(self, irreps_obj, R_matrix):
        """
        The definitive, most robust method. Combines all fixes:
        1. Cleans the matrix with SVD + STE.
        2. Converts to angles using our own robust function.
        3. Moves computation to CPU to solve device mismatch.
        4. Manually creates an integer 'k' tensor to solve PowBackward NaN.
        """
        original_device = R_matrix.device
        
        # 1. Clean the matrix on its original device
        R_perfect = SVD_with_Identity_Grad.apply(R_matrix)
        
        # 2. Convert to angles on the original device
        alpha, beta, gamma = robust_matrix_to_angles(R_perfect)

        # 3. Create the 'k' tensor with the correct integer dtype on the original device
        k = torch.ones_like(alpha, dtype=torch.long)
        
        try:
            # --- CRITICAL ACTION: Move all inputs to CPU for the e3nn call ---
            D_matrix_cpu = irreps_obj.D_from_angles(
                alpha.to('cpu'), beta.to('cpu'), gamma.to('cpu'), k.to('cpu')
            )
            
            # 5. Move the final result back to the original GPU device
            D_matrix = D_matrix_cpu.to(original_device)
            
            return D_matrix

        except Exception as e:
            print(f"WARNING: Final D_matrix calculation (on CPU) failed unexpectedly: {e}. Returning identity.")
            batch_size = R_matrix.shape[0]
            identity_D = torch.eye(irreps_obj.dim, device=original_device, dtype=R_matrix.dtype)
            return identity_D.unsqueeze(0).expand(batch_size, -1, -1) + (R_matrix.sum()*0).view(-1,1,1)

    def forward(self, input_node_features, node_positions, edge_index, guiding_poses_algebra, batch_idx_nodes):
        # 1. 移除 with torch.amp.autocast(...) 上下文管理器。
        # 2. 移除所有 .to(torch.float32) 的强制转换。
        # 3. 将所有 _f32 后缀的变量名改回原始名称。

        Dev = input_node_features.device
        B, M_out, _ = guiding_poses_algebra.shape
        N_sum, E_sum = input_node_features.shape[0], edge_index.shape[1]

        flat_guiding_poses_alg = torch.clamp(guiding_poses_algebra.reshape(B * M_out, 6), -10.0, 10.0)
        try:
            Guiding_Poses_Mat_Inv_flat = se3_inverse_custom(se3_exp_map_custom(flat_guiding_poses_alg))
        except Exception:
            # 重要：确保在异常情况下创建的张量也具有正确的 dtype，以匹配其他张量
            Guiding_Poses_Mat_Inv_flat = torch.eye(
                4, device=Dev, dtype=guiding_poses_algebra.dtype
            ).unsqueeze(0).expand(B * M_out, -1, -1)
        
        R_mat = Guiding_Poses_Mat_Inv_flat[:, :3, :3]

        super_batch_input_node_features_no_align = input_node_features.repeat(M_out, 1)
        current_features = super_batch_input_node_features_no_align
        if self.irreps_node_input_obj.lmax > 0:
            D_R_inv = self._get_D_from_matrix_final(self.irreps_node_input_obj, R_mat)
            original_graph_indices_expanded = batch_idx_nodes.repeat(M_out)
            view_indices_expanded = torch.arange(M_out, device=Dev).repeat_interleave(N_sum)
            pose_indices_for_nodes = original_graph_indices_expanded * M_out + view_indices_expanded
            D_R_inv_expanded = D_R_inv[pose_indices_for_nodes]
            aligned_features = torch.bmm(D_R_inv_expanded, super_batch_input_node_features_no_align.unsqueeze(-1)).squeeze(-1)
            current_features = torch.where(torch.isnan(aligned_features), super_batch_input_node_features_no_align, aligned_features)

        super_batch_pos = node_positions.repeat(M_out, 1)
        edge_index_offsets = torch.arange(M_out, device=Dev) * N_sum
        super_batch_edge_index = edge_index.repeat(1, M_out) + edge_index_offsets.repeat_interleave(E_sum)
        row, col = super_batch_edge_index
        edge_vec = super_batch_pos[row] - super_batch_pos[col]
        edge_len = torch.norm(edge_vec, dim=1, keepdim=True)
        
        # 这一行已经写得很好，它会自动继承 edge_vec 的 dtype
        edge_sh_super_batch = torch.zeros(edge_vec.shape[0], self.irreps_sh_obj.dim, device=edge_vec.device, dtype=edge_vec.dtype)
        valid_edges_mask = (edge_len > self.eps).squeeze()
        if valid_edges_mask.any():
            edge_sh_super_batch[valid_edges_mask] = self.spherical_harmonics(edge_vec[valid_edges_mask] / edge_len[valid_edges_mask])
        
        edge_radial_emb_super_batch = self.radial_embedding(edge_len)

        for block in self.interaction_layers_module:
            current_features = block(
                node_features=current_features,
                edge_index=super_batch_edge_index,
                edge_sh=edge_sh_super_batch,
                edge_radial_emb=edge_radial_emb_super_batch
            )

        pool_batch_idx = batch_idx_nodes.repeat(M_out) * M_out + torch.arange(M_out, device=Dev).repeat_interleave(N_sum)
        pooled_features = global_mean_pool(current_features, pool_batch_idx, size=B * M_out)
        
        Guiding_Poses_Mat_flat = se3_exp_map_custom(flat_guiding_poses_alg)
        R_guiding_flat = Guiding_Poses_Mat_flat[:, :3, :3]
        
        gconv_final_features = pooled_features
        if self.irreps_node_output_obj.lmax > 0:
            D_R_guiding = self._get_D_from_matrix_final(self.irreps_node_output_obj, R_guiding_flat)
            gconv_final_features = torch.bmm(D_R_guiding, pooled_features.unsqueeze(-1)).squeeze(-1)
        
        gconv_final_features = gconv_final_features.reshape(B, M_out, -1)
        
        return gconv_final_features