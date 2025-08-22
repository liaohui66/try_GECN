# capsule_conv.py 

import torch
import torch.nn as nn
from e3nn import o3
from typing import List, Dict, Any, Tuple, Optional

from capsule_layer import SE3CapsuleLayer
from group_conv import SE3GroupConvLayer
from geometry_utils import se3_log_map_custom
try:
    from debug_utils import DummyCapsuleLayer
    DEBUG_MODE_AVAILABLE = True
except ImportError:
    DEBUG_MODE_AVAILABLE = False
    DummyCapsuleLayer = None

class SE3CapsuleConvBlock(nn.Module):
    def __init__(self,
                 num_main_iterations: int,
                 routing_layer_configs: List[Dict[str, Any]],
                 gconv_layer_configs: List[Dict[str, Any]],
                 base_gnn_output_irreps_str: str,
                 use_dummy_routing_for_debug: bool = False):
        super().__init__()
        assert num_main_iterations > 0
        assert len(routing_layer_configs) == num_main_iterations
        assert len(gconv_layer_configs) == num_main_iterations

        self.num_main_iterations = num_main_iterations
        self.routing_layers = nn.ModuleList()
        self.gconv_layers = nn.ModuleList()

        for i in range(num_main_iterations):
            # 1. 路由层
            r_config = routing_layer_configs[i]
            if use_dummy_routing_for_debug and DEBUG_MODE_AVAILABLE:
                self.routing_layers.append(DummyCapsuleLayer(**r_config))
            else:
                self.routing_layers.append(SE3CapsuleLayer(**r_config))

            # 2. 群卷积层
            gc_config = gconv_layer_configs[i].copy()
            gc_config["irreps_node_input"] = base_gnn_output_irreps_str
            self.gconv_layers.append(SE3GroupConvLayer(**gc_config))
            
    def forward(self,
                current_P_alg: torch.Tensor,
                current_A: torch.Tensor,
                base_node_features: torch.Tensor,
                base_node_positions: torch.Tensor,
                base_edge_index: torch.Tensor,
                base_batch_idx: torch.Tensor,
                current_F_caps: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        loop_P = current_P_alg
        loop_A = current_A
        loop_F = current_F_caps

        for k in range(self.num_main_iterations):
            routing_layer_k = self.routing_layers[k]
            gconv_layer_k = self.gconv_layers[k]
            
            P_routed, A_routed = routing_layer_k(loop_P, loop_A)
            
            F_gconv = gconv_layer_k(
                input_node_features=base_node_features,
                node_positions=base_node_positions,
                edge_index=base_edge_index,
                guiding_poses_algebra=P_routed,
                batch_idx_nodes=base_batch_idx
            )

            if loop_F.shape != F_gconv.shape:
                raise ValueError(
                    f"Shape mismatch for residual connection in iteration {k}: "
                    f"loop_F shape is {loop_F.shape}, but F_gconv shape is {F_gconv.shape}. "
                    "Please ensure capsule numbers and feature irreps are consistent across iterations in your config."
                )
            
            updated_F = loop_F + F_gconv

            loop_P = P_routed
            loop_A = A_routed
            loop_F = updated_F

        final_F_adjusted = loop_F * loop_A.unsqueeze(-1)

        return loop_P, loop_A, final_F_adjusted

if __name__ == '__main__':
    import unittest
    
    # --- 模拟配置 ---
    B, N_avg, M = 2, 10, 4 # Batch, Nodes, Capsules
    BASE_IRREPS_STR = "2x0e+1x1o"
    CAPS_IRREPS_STR = "4x0e+2x1o" 
    
    # 关键：为了测试残差连接，gconv的输出irreps必须和输入的胶囊irreps一致
    iter_config_for_test = {
        "num_main_iterations": 2,
        "routing_layer_configs": [
            {"num_in_capsules": M, "num_out_capsules": M, "num_routing_iterations": 2},
            {"num_in_capsules": M, "num_out_capsules": M, "num_routing_iterations": 2}
        ],
        "gconv_layer_configs": [
            {"irreps_node_output": CAPS_IRREPS_STR, "irreps_sh": "1x0e+1x1o", "num_basis_radial": 8, "radial_mlp_hidden_dims": [16]},
            {"irreps_node_output": CAPS_IRREPS_STR, "irreps_sh": "1x0e+1x1o", "num_basis_radial": 8, "radial_mlp_hidden_dims": [16]}
        ],
        "base_gnn_output_irreps_str": BASE_IRREPS_STR,
    }

    class TestResidualConnection(unittest.TestCase):
        def setUp(self):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # 使用 self.iter_block 实例化 SE3CapsuleConvBlock
            self.iter_block = SE3CapsuleConvBlock(**iter_config_for_test).to(self.device)

        def _generate_mock_data(self):
            P_in = torch.randn(B, M, 6, device=self.device)
            A_in = torch.rand(B, M, device=self.device)
            F_in = torch.randn(B, M, o3.Irreps(CAPS_IRREPS_STR).dim, device=self.device, requires_grad=True)
            
            base_nodes = torch.randn(B * N_avg, o3.Irreps(BASE_IRREPS_STR).dim, device=self.device)
            base_pos = torch.randn(B * N_avg, 3, device=self.device)
            
            edge_index_list, batch_idx_list = [], []
            for i in range(B):
                # 创建一个简单的边索引
                edge_index_list.append(torch.randint(0, N_avg, (2, N_avg * 2), device=self.device) + i * N_avg)
                batch_idx_list.append(torch.full((N_avg,), i, device=self.device))
            
            base_edges = torch.cat(edge_index_list, dim=1)
            base_batch = torch.cat(batch_idx_list, dim=0)
            
            return P_in, A_in, F_in, base_nodes, base_pos, base_edges, base_batch

        def test_gradient_flow_through_residual_path(self):
            print("\n--- Testing Gradient Flow Through Residual Path ---")
            
            print("Freezing all parameters in gconv_layers...")
            for gconv_layer in self.iter_block.gconv_layers:
                for param in gconv_layer.parameters():
                    param.requires_grad = False

            P_in, A_in, F_in, base_nodes, base_pos, base_edges, base_batch = self._generate_mock_data()
            
            # 使用 self.iter_block 调用 forward
            _, _, F_out = self.iter_block(P_in, A_in, base_nodes, base_pos, base_edges, base_batch, F_in)

            loss = F_out.sum()
            loss.backward()

            self.assertIsNotNone(F_in.grad, "Gradient of F_in should NOT be None if residual connection works.")
            grad_norm = F_in.grad.norm().item()
            print(f"Gradient norm of input F_in: {grad_norm:.4f}")
            self.assertGreater(grad_norm, 1e-8, "Gradient of F_in is effectively zero, residual path failed.")
            
            print("SUCCESS: Gradient successfully flowed back to F_in through the residual path.")

    # 运行测试
    unittest.main(argv=['first-arg-is-ignored'], exit=False)