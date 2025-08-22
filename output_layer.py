# output_layer.py

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.io import CartesianTensor
from typing import Optional

def piezoelectric_tensor_to_voigt_3x6(d_ijk_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of 3x3x3 Cartesian piezoelectric tensors (d_ijk)
    to 3x6 Voigt matrix form (d_im), assuming d_ijk = d_ikj symmetry.

    Args:
        d_ijk_tensor: Tensor of shape (B, 3, 3, 3) or (3, 3, 3).

    Returns:
        Tensor of shape (B, 3, 6) or (3, 6).
    """
    if d_ijk_tensor.ndim == 3: # Single tensor
        d_ijk_tensor = d_ijk_tensor.unsqueeze(0) # Add batch dim
        single_tensor_input = True
    elif d_ijk_tensor.ndim == 4:
        single_tensor_input = False
    else:
        raise ValueError("Input tensor must be of shape (3,3,3) or (B,3,3,3)")

    B = d_ijk_tensor.shape[0]
    d_voigt = torch.zeros(B, 3, 6, device=d_ijk_tensor.device, dtype=d_ijk_tensor.dtype)

    # d_im where i is first index (rows: 1,2,3) and m is Voigt index (cols: 1..6)
    # Standard Voigt mapping for d_ijk = d_ikj:
    # d_i1 = d_i11
    d_voigt[:, :, 0] = d_ijk_tensor[:, :, 0, 0]
    # d_i2 = d_i22
    d_voigt[:, :, 1] = d_ijk_tensor[:, :, 1, 1]
    # d_i3 = d_i33
    d_voigt[:, :, 2] = d_ijk_tensor[:, :, 2, 2]

    # d_i4 = d_i23 (or d_i32, should be same due to "ijk=ikj" formula if used)
    d_voigt[:, :, 3] = d_ijk_tensor[:, :, 1, 2] 
    # d_i5 = d_i13 (or d_i31)
    d_voigt[:, :, 4] = d_ijk_tensor[:, :, 0, 2] 
    # d_i6 = d_i12 (or d_i21)
    d_voigt[:, :, 5] = d_ijk_tensor[:, :, 0, 1] 
    
    if single_tensor_input:
        return d_voigt.squeeze(0)
    return d_voigt

class SE3CovariantOutputLayer(torch.nn.Module):
    def __init__(self,
                 input_feature_irreps_str: str,
                 cartesian_tensor_formula: str = "ijk=ikj",
                 # output_bias: bool = True
                ):
        super().__init__()
        self.input_feature_irreps = o3.Irreps(input_feature_irreps_str)
        self.cartesian_tensor_formula = cartesian_tensor_formula

        # 1. Instantiate CartesianTensor. It IS an Irreps object itself.
        self.to_cartesian_tensor = CartesianTensor(formula=self.cartesian_tensor_formula)
        
        # 2. The o3.Linear layer must output features with Irreps matching self.to_cartesian_tensor
        # Since CartesianTensor IS an Irreps object, we can use it directly.
        self.target_tensor_irreps_for_linear = self.to_cartesian_tensor
        
        print(f"SE3CovariantOutputLayer: CartesianTensor '{self.cartesian_tensor_formula}' "
              f"IS Irreps: {self.target_tensor_irreps_for_linear} " # Message reflects change
              f"(dim: {self.target_tensor_irreps_for_linear.dim})")

        self.projection_to_tensor_irreps = o3.Linear(
            self.input_feature_irreps,
            self.target_tensor_irreps_for_linear, 
            # bias=output_bias
        )

    def forward(self, aggregated_equivariant_feature_tensor: torch.Tensor) -> torch.Tensor:
        tensor_basis_features = self.projection_to_tensor_irreps(aggregated_equivariant_feature_tensor)
        cartesian_output = self.to_cartesian_tensor.to_cartesian(tensor_basis_features)
        
        return cartesian_output

# # --- Test Code ---
# if __name__ == '__main__':
#     import unittest

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     dtype = torch.float32
#     torch.manual_seed(42) # For reproducible random numbers
#     torch.set_default_dtype(dtype)


#     print("\n" + "="*40 + "\n--- DEMO: SE3CovariantOutputLayer Prediction ---" + "\n" + "="*40)

#     # 1. Define simulated input aggregated feature's Irreps and dimension
#     agg_feat_irreps_str = "2x0e+2x1o+1x2e" # Example from previous tests
#     agg_feat_irreps = o3.Irreps(agg_feat_irreps_str)
#     agg_feat_dim = agg_feat_irreps.dim
#     print(f"Simulated Aggregated Feature Irreps: {agg_feat_irreps_str} (dim: {agg_feat_dim})")

#     # 2. Instantiate SE3CovariantOutputLayer
#     # Predict d_ijk with d_ijk = d_ikj symmetry
#     output_layer_formula = "ijk=ikj"
#     output_layer = SE3CovariantOutputLayer(
#         input_feature_irreps_str=agg_feat_irreps_str,
#         cartesian_tensor_formula=output_layer_formula
#     ).to(device) # Removed dtype=dtype, as Linear layers will infer from input
#     output_layer.eval()
#     print(f"Instantiated SE3CovariantOutputLayer with formula: '{output_layer_formula}'")


#     # 3. Create a batch of simulated input aggregated features
#     batch_size_demo = 2
#     mock_aggregated_features = torch.randn(batch_size_demo, agg_feat_dim, device=device, dtype=dtype)
#     print(f"\nSimulated input aggregated_features shape: {mock_aggregated_features.shape}")

#     # 4. Get predictions
#     with torch.no_grad():
#         predicted_piezo_tensors_cartesian = output_layer(mock_aggregated_features) # (B, 3, 3, 3)
    
#     # 5. Display results
#     print(f"\nShape of predicted Cartesian tensors (d_ijk): {predicted_piezo_tensors_cartesian.shape}")

#     for b in range(batch_size_demo):
#         print(f"\n--- Predicted Piezoelectric Tensor for Sample {b} ---")
        
#         current_cartesian_tensor = predicted_piezo_tensors_cartesian[b]
#         print("Full Cartesian d_ijk (rounded to 4 decimal places):")
#         # Print in a more readable format, e.g., slices
#         for i in range(3):
#             print(f"d_{i+1}jk slice:\n{torch.round(current_cartesian_tensor[i] * 10000) / 10000}") # Rounds to 4 decimal places

#         # Verify symmetry d_ijk = d_ikj for this formula
#         print("\nVerifying symmetry d_ijk = d_ikj (example d_123 vs d_132):")
#         print(f"  d_123: {current_cartesian_tensor[0, 1, 2].item():.4f}")
#         print(f"  d_132: {current_cartesian_tensor[0, 2, 1].item():.4f}")
#         assert torch.allclose(current_cartesian_tensor[0,1,2], current_cartesian_tensor[0,2,1], atol=1e-6), "Symmetry d_123=d_132 failed"

#         # Convert to Voigt 3x6 matrix and print
#         voigt_matrix = piezoelectric_tensor_to_voigt_3x6(current_cartesian_tensor)
#         print(f"\nVoigt matrix (3x6) representation for Sample {b} (rounded to 4 decimal places):")
#         print(torch.round(voigt_matrix * 10000) / 10000)
        
#     print("\n" + "="*40 + "\n--- DEMO Finished ---" + "\n" + "="*40)

#     # --- You can still run your unittests after the demo ---
#     print("\n\n" + "="*40 + "\n--- RUNNING UNIT TESTS ---" + "\n" + "="*40)

# unittest.main(argv=['first-arg-is-ignored'], exit=False)
# print("\n" + "="*30 + "\n--- SE3CovariantOutputLayer Tests Finished ---" + "\n" + "="*30)