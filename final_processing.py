# final_processing.py
import torch
import torch.nn as nn
from e3nn import o3
from typing import Tuple, Optional
import logging
logger = logging.getLogger(__name__)

# Assuming geometry_utils might be needed if more complex processing is added
# For now, for weighted average, it might not be directly needed by this class
try:
    from .geometry_utils import se3_exp_map_custom, se3_log_map_custom # For testing
except ImportError:
    from geometry_utils import se3_exp_map_custom, se3_log_map_custom # For testing


class FinalCapsuleProcessor(nn.Module):
    def __init__(self,
                 input_capsule_feature_irreps_str: str,
                 normalize_aggregation_weights: bool = True,
                 eps: float = 1e-8):
        super().__init__()
        self.input_capsule_feature_irreps = o3.Irreps(input_capsule_feature_irreps_str)
        self.normalize_aggregation_weights = normalize_aggregation_weights
        self.eps = eps

    def forward(self,
                final_P_alg: torch.Tensor,
                final_A: torch.Tensor,
                final_F_caps_tensor: torch.Tensor
               ) -> torch.Tensor: # Changed return type annotation
        logger.info("Executing FinalCapsuleProcessor with numerically stable weight aggregation.")
        B, M_N, C_F_N_dim = final_F_caps_tensor.shape
        # Ensure the input tensor's last dimension matches the irreps dimension
        if C_F_N_dim != self.input_capsule_feature_irreps.dim:
            raise ValueError(f"Input final_F_caps_tensor last dimension ({C_F_N_dim}) "
                             f"does not match input_capsule_feature_irreps dimension ({self.input_capsule_feature_irreps.dim})")

        weights = final_A.clamp(min=0) 

        if self.normalize_aggregation_weights:
            sum_weights = weights.sum(dim=1, keepdim=True).clamp(min=self.eps)
            normalized_weights = weights / sum_weights
        else:
            normalized_weights = weights # Use raw activations as weights if not normalizing

        weights_expanded = normalized_weights.unsqueeze(-1)
        weighted_features_tensor = final_F_caps_tensor * weights_expanded
        aggregated_feature_tensor = weighted_features_tensor.sum(dim=1) # (B, C_F_N_dim)
        
        # The output is now just the tensor. Its Irreps is self.input_capsule_feature_irreps.
        return aggregated_feature_tensor

# --- Test Code ---
if __name__ == '__main__':
    import unittest

    def generate_random_se3_matrix_for_test(device, dtype=torch.float32): # Renamed to avoid conflict
        R = o3.rand_matrix().to(dtype=dtype, device=device)
        t = torch.randn(3, device=device, dtype=dtype) * 0.1
        T_mat = torch.eye(4, device=device, dtype=dtype)
        T_mat[:3, :3] = R
        T_mat[:3, 3] = t
        return T_mat

    class TestFinalCapsuleProcessor(unittest.TestCase):
        def setUp(self):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.dtype = torch.float32
            print(f"\n--- Test Case: {self.id()} on device {self.device} ---")

            self.B = 2
            self.M_N = 5 # Number of final capsules
            self.caps_feat_irreps_str = "2x0e+1x1o+1x2e" # Example Irreps
            self.caps_feat_irreps = o3.Irreps(self.caps_feat_irreps_str)
            self.C_F_N_dim = self.caps_feat_irreps.dim

            self.processor = FinalCapsuleProcessor(
                input_capsule_feature_irreps_str=self.caps_feat_irreps_str,
                normalize_aggregation_weights=True
            ).to(self.device, dtype=self.dtype)
            self.processor.eval()

        def _create_mock_inputs(self):
            # final_P_alg: (B, M_N, 6)
            final_P_alg = torch.randn(self.B, self.M_N, 6, device=self.device, dtype=self.dtype) * 0.1
            # final_A: (B, M_N)
            final_A = torch.rand(self.B, self.M_N, device=self.device, dtype=self.dtype)
            # final_F_caps_tensor: (B, M_N, C_F_N_dim)
            final_F_caps_tensor = torch.randn(self.B, self.M_N, self.C_F_N_dim, device=self.device, dtype=self.dtype)
            return final_P_alg, final_A, final_F_caps_tensor

        def test_01_forward_pass_and_shapes(self):
            print("Testing forward pass and output shapes...")
            P_in, A_in, F_in_tensor = self._create_mock_inputs()
            
            # Processor now returns a tensor
            agg_feat_tensor = self.processor(P_in, A_in, F_in_tensor)

            # self.assertIsInstance(agg_feat_ir, o3.IrrepsArray, "Output should be IrrepsArray") # REMOVED
            # The processor's output irreps is implicitly self.processor.input_capsule_feature_irreps
            self.assertEqual(agg_feat_tensor.shape, (self.B, self.C_F_N_dim), "Output shape mismatch")

            self.assertFalse(torch.isnan(agg_feat_tensor).any(), "NaN in output")
            self.assertFalse(torch.isinf(agg_feat_tensor).any(), "Inf in output")
            print("Forward pass, shapes, and basic value checks OK.")

        def test_02_se3_equivariance(self, atol=1e-5):
            print("Testing SE(3) equivariance of aggregated feature...")
            P_orig_alg, A_orig, F_orig_caps_tensor = self._create_mock_inputs()

            # 1. Output with original inputs
            agg_feat_orig_tensor = self.processor(P_orig_alg, A_orig, F_orig_caps_tensor) # Now a tensor

            # 2. Create random SE(3) transformations
            G_matrices_batch = torch.stack(
                [generate_random_se3_matrix_for_test(self.device, self.dtype) for _ in range(self.B)], dim=0
            )
            R_matrices_batch = G_matrices_batch[:, :3, :3]

            # 3. Transform inputs (P_transformed_input_alg, A_transformed_input, F_transformed_input_caps_tensor)
            # ... (Pose and Activation transformation logic remains the same) ...
            T_orig_matrices = se3_exp_map_custom(P_orig_alg.reshape(-1, 6)).reshape(self.B, self.M_N, 4, 4)
            T_transformed_input_matrices = torch.matmul(G_matrices_batch.unsqueeze(1), T_orig_matrices)
            P_transformed_input_alg = se3_log_map_custom(T_transformed_input_matrices.reshape(-1, 4, 4)).reshape(self.B, self.M_N, 6)
            A_transformed_input = A_orig 
            F_transformed_input_caps_tensor_list = []
            for b_idx in range(self.B):
                D_R_caps_feat_b = self.caps_feat_irreps.D_from_matrix(R_matrices_batch[b_idx])
                transformed_F_b_tensor = F_orig_caps_tensor[b_idx] @ D_R_caps_feat_b.T
                F_transformed_input_caps_tensor_list.append(transformed_F_b_tensor)
            F_transformed_input_caps_tensor = torch.stack(F_transformed_input_caps_tensor_list, dim=0)


            # 4. Output with transformed inputs
            agg_feat_transformed_input_tensor = self.processor( # Now a tensor
                P_transformed_input_alg, A_transformed_input, F_transformed_input_caps_tensor
            )

            # 5. Calculate expected transformed aggregated feature
            # agg_feat_orig_tensor: (B, C_F_N_dim)
            # R_matrices_batch: (B, 3, 3)
            agg_feat_expected_transformed_list = []
            for b_idx in range(self.B):
                # The Irreps for agg_feat_orig_tensor is self.caps_feat_irreps
                D_R_agg_feat_b = self.caps_feat_irreps.D_from_matrix(R_matrices_batch[b_idx])
                orig_tensor_b = agg_feat_orig_tensor[b_idx] # (C_F_N_dim)
                transformed_tensor_b = orig_tensor_b @ D_R_agg_feat_b.T # (C_F_N_dim)
                agg_feat_expected_transformed_list.append(transformed_tensor_b)
            agg_feat_expected_transformed_tensor = torch.stack(agg_feat_expected_transformed_list, dim=0)
            
            diff = torch.norm(agg_feat_transformed_input_tensor - agg_feat_expected_transformed_tensor)
            self.assertTrue(
                torch.allclose(agg_feat_transformed_input_tensor, agg_feat_expected_transformed_tensor, atol=atol),
                f"Aggregated feature not SE(3) equivariant. Diff norm: {diff.item():.3e}"
            )
            print(f"Aggregated feature SE(3) equivariance check passed (Diff: {diff.item():.3e}).")

    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print("\n" + "="*30 + "\n--- FinalCapsuleProcessor Tests Finished ---" + "\n" + "="*30)