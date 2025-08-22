# primary_capsule.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from torch_scatter import scatter_mean
from typing import Tuple, Optional


class CapsulePoseActivationHead(nn.Module): # Keep this definition as corrected before
    def __init__(self,
                 input_features_irreps: str,
                 activation_hidden_dim: int = 32,
                 debug: bool = False):
        super().__init__()
        self.input_features_irreps_obj = o3.Irreps(input_features_irreps)
        self.debug = debug
        self.scalar_features_for_act_irreps = o3.Irreps(
            [(mul, ir) for mul, ir in self.input_features_irreps_obj if ir.l == 0]
        )
        activation_mlp_input_dim = self.scalar_features_for_act_irreps.dim
        if activation_mlp_input_dim > 0:
            self.activation_mlp = nn.Sequential(
                nn.Linear(activation_mlp_input_dim, activation_hidden_dim),
                nn.SiLU(),
                nn.Linear(activation_hidden_dim, 1),
                nn.Sigmoid()
            )
        else:
            self.activation_mlp = None
        self.pose_translation_projection = o3.Linear(
            self.input_features_irreps_obj, o3.Irreps("1x1o"), internal_weights=True
        )
        self.pose_rotation_projection = o3.Linear(
            self.input_features_irreps_obj, o3.Irreps("1x1o"), internal_weights=True
        )

    def forward(self, aggregated_caps_feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # --- Step 1: Calculate activation (logic remains the same) ---
            activation = None
            if self.activation_mlp is not None and self.scalar_features_for_act_irreps.dim > 0:
                scalar_data = aggregated_caps_feature.narrow(1, 0, self.scalar_features_for_act_irreps.dim)
                activation = self.activation_mlp(scalar_data)
            else:
                num_entities = aggregated_caps_feature.shape[0]
                activation = torch.sigmoid(torch.zeros(num_entities, 1, device=aggregated_caps_feature.device))
            
            activation = activation.squeeze(-1)

            # --- Step 2: Project to raw translation and rotation vectors ---
            raw_translation_vector = self.pose_translation_projection(aggregated_caps_feature)
            raw_rotation_vector = self.pose_rotation_projection(aggregated_caps_feature)

            # --- Step 3: [CRITICAL ACTION] Stabilize outputs using tanh ---
            # This constrains all components of the algebra vectors to [-1, 1].
            # For the rotation vector omega, this means its norm ||omega|| is bounded,
            # preventing it from reaching the singularity at ||omega|| = pi for the
            # exponential map. This is the core fix.
            stable_translation_vector = torch.tanh(raw_translation_vector)
            stable_rotation_vector = torch.tanh(raw_rotation_vector)
            
            # --- Step 4: Concatenate the STABILIZED vectors to form the 6D pose algebra ---
            pose_algebra = torch.cat([stable_translation_vector, stable_rotation_vector], dim=-1)
            
            return pose_algebra, activation


class SE3PrimaryCapsule(nn.Module): # V3: GAP then Decode
    def __init__(self,
                 gnn_output_irreps: str,      # Irreps of node_equivariant_features from GNN
                 num_primary_capsules: int,   # M_primary
                 # Irreps for the feature of each decoded primary capsule before pose/act head
                 irreps_decoded_capsule_feature: Optional[str] = None,
                 capsule_head_activation_hidden_dim: int = 32,
                 use_separate_decoders: bool = True, # True: M_primary Linear layers, False: 1 Linear then split
                 debug: bool = False
                ):
        super().__init__()
        self.gnn_output_irreps_obj = o3.Irreps(gnn_output_irreps)
        self.num_primary_capsules = num_primary_capsules
        
        if irreps_decoded_capsule_feature is None:
            # Default to using the GNN output irreps for the decoded capsule features
            self.irreps_decoded_capsule_feature_obj = self.gnn_output_irreps_obj
        else:
            self.irreps_decoded_capsule_feature_obj = o3.Irreps(irreps_decoded_capsule_feature)
        
        self.use_separate_decoders = use_separate_decoders
        self.debug = debug

        if self.debug:
            print(f"[DEBUG PrimaryCapsV3 Init] GNN Output Irreps: {self.gnn_output_irreps_obj}")
            print(f"[DEBUG PrimaryCapsV3 Init] Num Primary Capsules: {self.num_primary_capsules}")
            print(f"[DEBUG PrimaryCapsV3 Init] Decoded Capsule Feature Irreps: {self.irreps_decoded_capsule_feature_obj}")
            print(f"[DEBUG PrimaryCapsV3 Init] Using Separate Decoders: {self.use_separate_decoders}")

        # 1. Decoder(s) to get M_primary capsule features from global_graph_feature
        if self.use_separate_decoders:
            self.capsule_feature_decoders = nn.ModuleList()
            for _ in range(self.num_primary_capsules):
                self.capsule_feature_decoders.append(
                    o3.Linear(self.gnn_output_irreps_obj, # Input: global_graph_feature
                              self.irreps_decoded_capsule_feature_obj, # Output: feature for one capsule
                              internal_weights=True)
                )
        else: # Single Linear layer then split/view
            self.combined_capsule_decoder = o3.Linear(
                self.gnn_output_irreps_obj,
                self.irreps_decoded_capsule_feature_obj * self.num_primary_capsules, # Output M_primary * caps_feat_dim
                internal_weights=True
            )
        if self.debug: print(f"Decoder(s) initialized.")

        # 2. Capsule Pose & Activation Head (common for all decoded capsule features)
        self.capsule_head = CapsulePoseActivationHead(
            input_features_irreps=str(self.irreps_decoded_capsule_feature_obj),
            activation_hidden_dim=capsule_head_activation_hidden_dim,
            debug=self.debug
        )
        if self.debug: print(f"CapsulePoseActivationHead initialized for V3.")


    def forward(self,
                node_equivariant_features: torch.Tensor, # (N_total_nodes, gnn_dim)
                gnn_output_irreps_obj_runtime: o3.Irreps,
                batch_idx_nodes: torch.Tensor             # (N_total_nodes,)
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            node_equivariant_features: SE(3) equivariant features for N_total_nodes.
            gnn_output_irreps_obj_runtime: Irreps object for node_equivariant_features.
            batch_idx_nodes: Tensor indicating graph membership for each node.

        Returns:
            primary_capsules_poses_algebra: (B, M_primary, 6)
            primary_capsules_activations: (B, M_primary)
            primary_capsules_features_decoded: (B, M_primary, decoded_capsule_feature_dim)
        """
        assert gnn_output_irreps_obj_runtime == self.gnn_output_irreps_obj, "Irreps mismatch"
        Dev = node_equivariant_features.device
        
        if batch_idx_nodes is None:
            B = 1
            batch_idx_nodes = torch.zeros(node_equivariant_features.shape[0], dtype=torch.long, device=Dev)
        else:
            B = batch_idx_nodes.max().item() + 1

        if self.debug:
            print(f"\n--- [DEBUG PrimaryCapsV3 Forward] ---")
            print(f"Input node_equivariant_features shape: {node_equivariant_features.shape}")
            print(f"Effective Batch Size B: {B}")

        # 1. Global Average Pooling of node features per graph
        # global_graph_feature: (B, gnn_dim) - This is SE(3) Equivariant
        global_graph_feature = scatter_mean(node_equivariant_features, batch_idx_nodes, dim=0, dim_size=B)
        if self.debug: print(f"Global graph feature shape after GAP: {global_graph_feature.shape}")

        # 2. Decode M_primary capsule features from the global_graph_feature
        # decoded_caps_features_list will store features for each primary capsule across batch
        # Each item in list: (B, decoded_caps_feat_dim)
        
        if self.use_separate_decoders:
            decoded_caps_features_list = []
            for i in range(self.num_primary_capsules):
                # self.capsule_feature_decoders[i] takes (B, gnn_dim) -> (B, decoded_caps_feat_dim)
                decoded_caps_features_list.append(self.capsule_feature_decoders[i](global_graph_feature))
            # Stack to get (B, M_primary, decoded_caps_feat_dim)
            primary_capsules_features_decoded = torch.stack(decoded_caps_features_list, dim=1)
        else:
            # combined_output: (B, M_primary * decoded_caps_feat_dim)
            combined_output = self.combined_capsule_decoder(global_graph_feature)
            primary_capsules_features_decoded = combined_output.view(
                B, self.num_primary_capsules, self.irreps_decoded_capsule_feature_obj.dim
            )
        
        # primary_capsules_features_decoded is SE(3) EQUIVARIANT
        if self.debug: print(f"Decoded primary capsule features shape: {primary_capsules_features_decoded.shape}")
        
        # 3. Extract Pose and Activation for each primary capsule
        # Reshape for capsule_head: (B * M_primary, decoded_caps_feat_dim)
        B_agg, M_agg, D_caps_feat = primary_capsules_features_decoded.shape # M_agg is M_primary
        flat_decoded_caps_features = primary_capsules_features_decoded.reshape(B_agg * M_agg, D_caps_feat)

        primary_caps_poses_alg_flat, primary_caps_acts_flat = self.capsule_head(flat_decoded_caps_features)
        
        primary_capsules_poses_algebra = primary_caps_poses_alg_flat.reshape(B_agg, M_agg, -1) # (B, M_primary, 6)
        primary_capsules_activations = primary_caps_acts_flat.reshape(B_agg, M_agg)       # (B, M_primary)

        if self.debug:
            print(f"Output primary_capsules_poses_algebra shape: {primary_capsules_poses_algebra.shape}")
            print(f"Output primary_capsules_activations shape: {primary_capsules_activations.shape}")
            print(f"--- [DEBUG PrimaryCapsV3 Forward END] ---\n")

        return primary_capsules_poses_algebra, primary_capsules_activations, primary_capsules_features_decoded
