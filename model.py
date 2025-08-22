# model.py
import torch
import torch_geometric
import torch.nn as nn
from torch_geometric.data import Data, Batch
from typing import List, Dict, Any, Optional
from e3nn import o3
from torch.utils.checkpoint import checkpoint

try:
    from encoder import SE3InvariantGraphEncoder
    from primary_capsule import SE3PrimaryCapsule
    from capsule_conv import SE3CapsuleConvBlock
    from final_processing import FinalCapsuleProcessor
    from output_layer import SE3CovariantOutputLayer
    from debug_utils import DummyEncoder, DummyIterativeBlock
    DEBUG_MODE_AVAILABLE = True # 标记调试模块可用
except ImportError:
    from .encoder import SE3InvariantGraphEncoder
    from .primary_capsule import SE3PrimaryCapsule
    from .capsule_conv import SE3CapsuleConvBlock
    from .final_processing import FinalCapsuleProcessor
    from .output_layer import SE3CovariantOutputLayer
    from .debug_utils import DummyEncoder, DummyIterativeBlock
    DEBUG_MODE_AVAILABLE = True

class EndToEndPiezoNet(nn.Module):
    def __init__(self,
                 gnn_config: Dict[str, Any],
                 primary_capsule_config: Dict[str, Any],
                 iterative_block_config: Dict[str, Any],
                 final_processor_config: Dict[str, Any],
                 output_layer_config: Dict[str, Any],
                ):
        super().__init__()
        
        self.base_gnn_output_irreps_str = gnn_config["irreps_node_output"]
        self.base_gnn_output_irreps_obj = o3.Irreps(self.base_gnn_output_irreps_str)
        
        # 1. Encoder 实例化
        self.encoder = SE3InvariantGraphEncoder(**gnn_config)
        print(f"EndToEndNet: Encoder initialized. Output Irreps: {self.base_gnn_output_irreps_str}")

        # 2. Primary Capsule Generator 实例化
        current_primary_capsule_config = primary_capsule_config.copy()
        current_primary_capsule_config["gnn_output_irreps"] = self.base_gnn_output_irreps_str
        self.primary_caps_generator = SE3PrimaryCapsule(**current_primary_capsule_config)
        self.M_0 = primary_capsule_config["num_primary_capsules"]
        self.irreps_F_0_str = primary_capsule_config.get("irreps_decoded_capsule_feature", self.base_gnn_output_irreps_str)
        self.irreps_F_0_obj = o3.Irreps(self.irreps_F_0_str)
        print(f"EndToEndNet: PrimaryCapsuleGenerator initialized. Outputting M_0={self.M_0} capsules. Feature Irreps: {self.irreps_F_0_str}")

        # 3. Iterative Block 实例化
        current_iterative_block_config = iterative_block_config.copy()
        current_iterative_block_config["base_gnn_output_irreps_str"] = self.base_gnn_output_irreps_str
        first_routing_config = current_iterative_block_config['routing_layer_configs'][0]
        if first_routing_config.get('num_in_capsules') != self.M_0:
            print(f"Warning: Adjusting iterative_block's first routing layer to match M_0={self.M_0}.")
            first_routing_config['num_in_capsules'] = self.M_0
        self.iterative_block = SE3CapsuleConvBlock(**current_iterative_block_config)
        self.irreps_F_N_str = iterative_block_config["gconv_layer_configs"][-1]["irreps_node_output"]
        self.irreps_F_N_obj = o3.Irreps(self.irreps_F_N_str)
        print(f"EndToEndNet: IterativeBlock initialized. Final Capsule Feature Irreps: {self.irreps_F_N_str}")

        # 4. Final Processor 实例化
        current_final_processor_config = final_processor_config.copy()
        current_final_processor_config["input_capsule_feature_irreps_str"] = self.irreps_F_N_str
        self.final_processor = FinalCapsuleProcessor(**current_final_processor_config)
        print(f"EndToEndNet: FinalProcessor initialized. Aggregated Feature Irreps: {self.irreps_F_N_str}")

        # 5. Output Layer 实例化
        current_output_layer_config = output_layer_config.copy()
        current_output_layer_config["input_feature_irreps_str"] = self.irreps_F_N_str
        self.output_layer = SE3CovariantOutputLayer(**current_output_layer_config)
        print(f"EndToEndNet: OutputLayer initialized.")

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        # 1. Encoder
        base_node_features_tensor, base_node_features_irreps = self.encoder(data)
        if torch.isnan(base_node_features_tensor).any():
            raise RuntimeError("NaN detected in Encoder output!")

        # 2. Primary Capsule Generator
        P_initial_alg, A_initial, F_initial_caps_tensor = self.primary_caps_generator(
            node_equivariant_features=base_node_features_tensor,
            gnn_output_irreps_obj_runtime=base_node_features_irreps,
            batch_idx_nodes=data.batch
        )
        if torch.isnan(P_initial_alg).any():
            raise RuntimeError("NaN detected in Primary Capsule Poses (P_initial_alg)!")
        if torch.isnan(A_initial).any():
            raise RuntimeError("NaN detected in Primary Capsule Activations (A_initial)!")
        if torch.isnan(F_initial_caps_tensor).any():
            raise RuntimeError("NaN detected in Primary Capsule Features (F_initial_caps_tensor)!")

        # 3. Iterative Convolutional Block
        final_P, final_A, final_F_caps_tensor = self.iterative_block(
            current_P_alg=P_initial_alg,
            current_A=A_initial,
            base_node_features=base_node_features_tensor,
            base_node_positions=data.pos,
            base_edge_index=data.edge_index,
            base_batch_idx=data.batch,
            current_F_caps=F_initial_caps_tensor
        )
        if torch.isnan(final_P).any():
            raise RuntimeError("NaN detected in Iterative Block Poses (final_P)!")
        if torch.isnan(final_A).any():
            raise RuntimeError("NaN detected in Iterative Block Activations (final_A)!")
        if torch.isnan(final_F_caps_tensor).any():
            raise RuntimeError("NaN detected in Iterative Block Features (final_F_caps_tensor)!")

        # 4. Final Capsule Processor
        aggregated_feature_tensor = self.final_processor(
            final_P, 
            final_A, 
            final_F_caps_tensor 
        )
        if torch.isnan(aggregated_feature_tensor).any():
            raise RuntimeError("NaN detected in Final Processor output!")

        # 5. Output Layer
        predicted_piezo_cartesian = self.output_layer(aggregated_feature_tensor)
        if torch.isnan(predicted_piezo_cartesian).any():
            raise RuntimeError("NaN detected in the final Output Layer!")
        
        return predicted_piezo_cartesian

# --- Example Usage and Basic Test ---
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)

    # --- Define Example Configurations (as per previous discussion) ---
    gnn_config_ex = {
        "num_atom_types": 90, "embedding_dim_scalar": 16, 
        "irreps_node_hidden": "16x0e+8x1o", "irreps_node_output": "8x0e+4x1o", 
        "irreps_edge_attr": "1x0e",  # <--- MODIFIED: To match the original error's DEBUG output
        "irreps_sh": "1x0e+1x1o", "max_radius": 3.0, 
        "num_basis_radial": 8, "radial_mlp_hidden": [32], "num_interaction_layers": 1, 
        "use_attention": False
    }
    base_gnn_output_irreps_for_example = gnn_config_ex["irreps_node_output"] 

    M_0_example = 4 
    irreps_F_0_example_str = "4x0e+2x1o" 
    primary_capsule_config_ex = {
        "num_primary_capsules": M_0_example,
        "irreps_decoded_capsule_feature": irreps_F_0_example_str,
        "capsule_head_activation_hidden_dim": 16, 
        "use_separate_decoders": True,
        "debug": False
    }

    num_main_iter_example = 1 
    M_N_example = 3 
    irreps_F_N_example_str = "2x0e+1x1o" 

    routing_cfgs_ex = [{
        "num_in_capsules": M_0_example, 
        "num_out_capsules": M_N_example, 
        "num_routing_iterations": 2 
    }]
    gconv_cfgs_ex = [{
        "irreps_node_output": irreps_F_N_example_str, 
        "irreps_sh": gnn_config_ex["irreps_sh"], 
        "num_basis_radial": gnn_config_ex["num_basis_radial"], 
        "radial_mlp_hidden_dims": gnn_config_ex["radial_mlp_hidden"],
        "num_interaction_layers": 1, 
        "interaction_block_type": "tfn"
    }]
    iterative_block_config_ex = {
        "num_main_iterations": num_main_iter_example,
        "routing_layer_configs": routing_cfgs_ex,
        "gconv_layer_configs": gconv_cfgs_ex,
    }
    
    final_processor_config_ex = {
        "normalize_aggregation_weights": True
    }

    output_layer_config_ex = {
        "cartesian_tensor_formula": "ijk=ikj"
    }

    print("\nInstantiating EndToEndPiezoNet with example configs...")
    try:
        model = EndToEndPiezoNet(
            gnn_config=gnn_config_ex,
            primary_capsule_config=primary_capsule_config_ex,
            iterative_block_config=iterative_block_config_ex,
            final_processor_config=final_processor_config_ex,
            output_layer_config=output_layer_config_ex
        ).to(device)
        print("EndToEndPiezoNet instantiated successfully!")
        
        print("\nCreating mock data and testing forward pass...")
        B_test = 2
        N_nodes_test_avg = 5 
        
        data_list = []
        for i in range(B_test):
            num_n = N_nodes_test_avg 
            pos = torch.randn(num_n, 3, device=device)
            x_atoms = torch.randint(0, gnn_config_ex["num_atom_types"], (num_n,), device=device) 
            
            src = torch.arange(num_n, device=device).repeat_interleave(num_n)
            dst = torch.arange(num_n, device=device).repeat(num_n)
            mask = src != dst
            edge_index = torch.stack([src[mask], dst[mask]], dim=0)
            if edge_index.shape[1] == 0 and num_n > 1: 
                edge_index = torch.tensor([[0],[1]], device=device, dtype=torch.long)
            elif num_n == 1: 
                 edge_index = torch.empty((2,0), device=device, dtype=torch.long)

            num_edges_current_graph = edge_index.shape[1]

            # --- ADDED: Create mock edge_attributes ---
            # Based on gnn_config_ex["irreps_edge_attr"] = "1x0e", dim is 1
            if num_edges_current_graph > 0:
                mock_edge_attributes = torch.randn(num_edges_current_graph, 1, device=device)
            else:
                mock_edge_attributes = torch.empty((0, 1), device=device)

            lattice = torch.eye(3, device=device).unsqueeze(0) 
            edge_shift = torch.zeros(edge_index.shape[1], 3, device=device)

            data_list.append(Data(x=x_atoms, pos=pos, edge_index=edge_index, 
                                  lattice=lattice, edge_shift=edge_shift,
                                  edge_attr=mock_edge_attributes # <--- ADDED: Pass edge_attr to Data
                                 ))
        
        batched_data = Batch.from_data_list(data_list).to(device)
        
        model.eval() 
        with torch.no_grad():
           prediction = model(batched_data)
           print(f"\nModel prediction shape: {prediction.shape}")
           assert prediction.shape == (B_test, 3, 3, 3), "Prediction shape is incorrect!"
           print("Forward pass successful and output shape is correct.")

    except Exception as e:
        print(f"Error during EndToEndPiezoNet instantiation or test: {e}")
        import traceback
        traceback.print_exc()