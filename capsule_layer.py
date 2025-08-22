# capsule_layer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .geometry_utils import (
        se3_exp_map_custom, 
        se3_log_map_custom, 
        se3_inverse_custom,
        se3_mean_algebra_custom,
        se3_distance_algebra_custom
    )
except ImportError:
    from geometry_utils import (
        se3_exp_map_custom, 
        se3_log_map_custom, 
        se3_inverse_custom,
        se3_mean_algebra_custom,
        se3_distance_algebra_custom
    )

class SE3CapsuleLayer(nn.Module):
    def __init__(self, 
                num_in_capsules: int,
                num_out_capsules: int,
                se3_algebra_dim: int = 6,
                num_routing_iterations: int = 3,
                learnable_transformations: bool = True, # <<< This is the input parameter
                mean_so3_method: str = "quaternion_nwa",
                distance_rot_weight: float = 1.0,
                ): 
        super().__init__()

        if se3_algebra_dim != 6:
            raise ValueError("se3_algebra_dim must be 6 for SE(3).")

        self.num_in_capsules = num_in_capsules
        self.num_out_capsules = num_out_capsules
        self.num_iterations = num_routing_iterations
        self.mean_so3_method = mean_so3_method
        self.distance_rot_weight = distance_rot_weight
        self.se3_dim = se3_algebra_dim

        self.learnable_transformations = learnable_transformations # Store the boolean flag

        if self.learnable_transformations: # Now using the instance attribute
            self.se3_transforms_tij_algebra = nn.Parameter(
                torch.randn(num_out_capsules, num_in_capsules, self.se3_dim) * 0.01 
            )
        else:
            self.se3_transforms_tij_algebra = None 
            # TODO: self.transform_generator_mlp = ... 

        self.routing_alpha = nn.Parameter(torch.ones(1, num_out_capsules, 1))
        self.routing_beta = nn.Parameter(torch.zeros(1, num_out_capsules, 1))
        self.activation_alpha = nn.Parameter(torch.ones(1, num_out_capsules, 1))
        self.activation_beta_minus_1 = nn.Parameter(torch.zeros(1, num_out_capsules, 1))

    def _apply_se3_transform_vote_algebra(self, p_i_algebra: torch.Tensor, t_ij_algebra: torch.Tensor) -> torch.Tensor:
        B = p_i_algebra.shape[0]
        M_out = self.num_out_capsules
        N_in = self.num_in_capsules 
        t_ij_algebra_on_device = t_ij_algebra.to(p_i_algebra.device)

        p_i_expanded = p_i_algebra.unsqueeze(1).expand(B, M_out, N_in, self.se3_dim)
        t_ij_expanded = t_ij_algebra_on_device.unsqueeze(0).expand(B, M_out, N_in, self.se3_dim) 

        flat_p_i = p_i_expanded.reshape(-1, self.se3_dim)
        flat_t_ij = t_ij_expanded.reshape(-1, self.se3_dim) 

        P_i_mat = se3_exp_map_custom(flat_p_i) 
        T_ij_mat = se3_exp_map_custom(flat_t_ij) 

        V_ij_mat = torch.matmul(P_i_mat, T_ij_mat)

        flat_v_ij_algebra = se3_log_map_custom(V_ij_mat)
        votes_algebra = flat_v_ij_algebra.reshape(B, M_out, N_in, self.se3_dim)
        return votes_algebra
    
    def forward(self, input_poses_algebra: torch.Tensor, input_activations: torch.Tensor):
        # 1. 移除 with torch.amp.autocast(...) 上下文和所有 .to(torch.float32) 调用
        #    直接使用原始的输入张量
        
        # 2. 将代码中所有对 _f32 变量的引用改回原始输入变量名
        B = input_poses_algebra.shape[0]
        if input_activations.ndim == 2:
            input_activations = input_activations.unsqueeze(-1)
        
        if self.se3_transforms_tij_algebra is None:
            raise NotImplementedError("Dynamic transform generation is not implemented.")
        
        # 直接使用 self.se3_transforms_tij_algebra，它应该与模型/输入设备和类型一致
        current_t_ij_algebra = self.se3_transforms_tij_algebra
        
        votes_algebra = self._apply_se3_transform_vote_algebra(input_poses_algebra, current_t_ij_algebra)

        # Initial pose proposal
        initial_routing_weights_expanded = input_activations.unsqueeze(1).expand(
            B, self.num_out_capsules, self.num_in_capsules, 1
        )
        current_output_poses_algebra = se3_mean_algebra_custom(
            votes_algebra, 
            initial_routing_weights_expanded,
            normalize_weights_method="sum"
        )

        # Iterative Routing
        for _ in range(self.num_iterations):
            pose1_for_dist = current_output_poses_algebra.unsqueeze(2).expand_as(votes_algebra)
            pose2_for_dist = votes_algebra
            
            original_shape_dist_input = pose1_for_dist.shape
            num_items_for_dist = original_shape_dist_input[0] * original_shape_dist_input[1] * original_shape_dist_input[2]
            
            flat_pose1_for_dist = pose1_for_dist.reshape(num_items_for_dist, self.se3_dim)
            flat_pose2_for_dist = pose2_for_dist.reshape(num_items_for_dist, self.se3_dim)

            flat_distances_delta = se3_distance_algebra_custom(
                flat_pose1_for_dist, 
                flat_pose2_for_dist,                            
                rot_weight=self.distance_rot_weight
            )
            distances_delta = flat_distances_delta.reshape(
                original_shape_dist_input[0], original_shape_dist_input[1], original_shape_dist_input[2]
            )

            agreement_scores = -distances_delta 
            routing_logits = self.routing_alpha * agreement_scores + self.routing_beta 
            routing_weights_w_unnormalized = torch.sigmoid(routing_logits)
            
            routing_weights_w = routing_weights_w_unnormalized * input_activations.transpose(1,2)
            
            current_output_poses_algebra = se3_mean_algebra_custom(
                votes_algebra,
                routing_weights_w.unsqueeze(-1),
                normalize_weights_method="sum"
            )

        # --- Final activation calculation ---
        final_pose1_expanded_for_dist = current_output_poses_algebra.unsqueeze(2).expand_as(votes_algebra)
        flat_final_pose1_for_dist = final_pose1_expanded_for_dist.reshape(num_items_for_dist, self.se3_dim)
        flat_final_distances_delta = se3_distance_algebra_custom(
            flat_final_pose1_for_dist, 
            flat_pose2_for_dist,
            rot_weight=self.distance_rot_weight
        )
        final_distances_delta = flat_final_distances_delta.reshape(
            original_shape_dist_input[0], original_shape_dist_input[1], original_shape_dist_input[2]
        )
        
        mean_distance_for_activation = final_distances_delta.mean(dim=-1)
        output_activations = torch.sigmoid(
            self.activation_alpha * (-mean_distance_for_activation.unsqueeze(-1)) + self.activation_beta_minus_1
        ) 
        output_activations = output_activations.squeeze(-1)

        return current_output_poses_algebra, output_activations