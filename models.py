# %%
import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from typing import Callable
# %%
class GNNModel(torch.nn.Module):
    def __init__(self, gnn_layers: list[MessagePassing], activation_functions: list[Callable], addDropOut: bool = False):
        super().__init__()
        torch.manual_seed(42)
        # self.conv1 = GCNConv(numNodeFeatures, hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.conv3 = GCNConv(hidden_channels, numClasses)
        self.gnn_layers = torch.nn.ModuleList(gnn_layers)
        self.activation_functions = activation_functions
        self.addDropOut = addDropOut
        if len(self.gnn_layers) != len(self.activation_functions):
            print("WARNING: layers are missing activation functions")
            print(f"{len(self.activation_functions) = }")
            print(f"{len(self.gnn_layers) = }")

    # return (softmax classifier results, hidden embeddings) 
    def forward(self, x, edge_index):
        # # First Message Passing Layer (Transformation)
        # out = self.conv1(x, edge_index)
        # out = F.tanh(out)
        # # out = F.dropout(out, p=0.5, training=self.training)
        # # Second Message Passing Layer
        # out = self.conv2(out, edge_index)
        # out = F.tanh(out)
        # # out = F.dropout(out, p=0.5, training=self.training)
        # out = self.conv3(out, edge_index)
        for idx, layer in enumerate(self.gnn_layers):
            x = layer(x, edge_index)
            x = self.activation_functions[idx](x)
            if self.addDropOut and idx < (len(self.gnn_layers) - 1):
                x = F.dropout(x, p=0.5, training=self.training)

        return F.log_softmax(x, dim=1), x