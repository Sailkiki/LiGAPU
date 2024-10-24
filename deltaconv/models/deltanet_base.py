import torch
from torch_geometric.nn import knn_graph

from ..nn import DeltaConv, MLP
from ..geometry.grad_div_mls import build_grad_div, build_tangent_basis, estimate_basis
    

class DeltaNetBase(torch.nn.Module):
    def __init__(self, in_channels, conv_channels, mlp_depth, num_neighbors, grad_regularizer, grad_kernel_width, centralize_first=True):

        super().__init__()
        self.k = num_neighbors
        self.grad_regularizer = grad_regularizer
        self.grad_kernel_width = grad_kernel_width
        conv_channels = [in_channels] + conv_channels
        self.convs = torch.nn.ModuleList()
        for i in range(len(conv_channels) - 1):
            last_layer = i == (len(conv_channels) - 2)
            self.convs.append(DeltaConv(conv_channels[i],
                                        conv_channels[i + 1],
                                        depth=mlp_depth,
                                        centralized=(centralize_first and i == 0),
                                        vector=not(last_layer)))


    def forward(self, data):
        # [n, 3]
        pos = data.pos
        # [n]
        batch = data.batch

        # K = 3 .[2, n*k]
        edge_index = knn_graph(pos, self.k, batch, loop=True, flow='target_to_source')


        if hasattr(data, 'norm') and data.norm is not None:
            normal = data.norm
            x_basis, y_basis = build_tangent_basis(normal)
        else:
            edge_index_normal = knn_graph(pos, 10, batch, loop=True, flow='target_to_source')
            # 使用 K-近邻图来估计法向量和切线基
            normal, x_basis, y_basis = estimate_basis(pos, edge_index_normal, orientation=pos)


        grad, div = build_grad_div(pos, normal, x_basis, y_basis, edge_index, batch, kernel_width=self.grad_kernel_width, regularizer=self.grad_regularizer)

        

        # x = data.x if hasattr(data, 'x') and data.x is not None else pos

        x = pos
        v = grad @ x


        out = []
        for conv in self.convs:
            x, v = conv(x, v, grad, div, edge_index)
            out.append(x)

        return out





