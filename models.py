# %%
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
# %%
class GCN(torch.nn.Module):
    def __init__(self, numNodeFeatures: int, hidden_channels: int, numClasses: int):
        super().__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(numNodeFeatures, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, numClasses)

    def forward(self, x, edge_index):
        # First Message Passing Layer (Transformation)
        out = self.conv1(x, edge_index)
        out = F.sigmoid(out)
        out = F.dropout(out, p=0.5, training=self.training)
        # Second Message Passing Layer
        out = self.conv2(out, edge_index)
        return F.log_softmax(out, dim=1)