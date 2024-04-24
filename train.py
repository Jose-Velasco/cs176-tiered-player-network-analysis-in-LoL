# %%
import torch
from pytorch_geometric_dataset import LOL_Dataset
# from torch.utils.data import SubsetRandomSampler
from torch_geometric.loader import DataLoader
# import numpy as np
from torch_geometric.transforms import NormalizeFeatures, Compose, ToDevice
# from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.loader import DataLoader
from train_utils import testPhase, trainPhase, CustomRandomNodeSplitMasker, CustomRandomNodeUnderSampler, ConcatNodeCentralities
from utils_visualize import visualize, visualize2
from models import GNNModel
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing, summary, GATConv
from typing import Callable
from utils import get_now_datetime
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

# %%
ROOT_DIR="PTG_data"
OUTPUT_DIR = "./experiments"
NUM_EPOCHS = 449
TEST_SIZE = 0.1
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42
BATCH_SIZE = 1
HIDDEN_CHANNELS = 16
torch.manual_seed(RANDOM_STATE)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
nodeUnderSample = CustomRandomNodeUnderSampler(RANDOM_STATE, allowSelfLoops=False)
nodeSPlitMasker = CustomRandomNodeSplitMasker(
    test_size=TEST_SIZE,
    validation_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE
)
concatNodeCentralitiesTransformer = ConcatNodeCentralities()
train_transformer = Compose(
    [
        ToDevice(DEVICE),
        NormalizeFeatures(),
        nodeSPlitMasker
    ]
)
train_val_test_pre_transforms = Compose(
    [
        nodeUnderSample,
        concatNodeCentralitiesTransformer
    ]
)
val_test_transformer = Compose(
    [
        ToDevice(DEVICE),
        NormalizeFeatures(),
        nodeSPlitMasker
    ]
)
print(f"{train_transformer = }")
print(f"{train_val_test_pre_transforms = }")
print(f"{val_test_transformer = }")

train_dataset = LOL_Dataset(
    root=ROOT_DIR,
    transform=train_transformer,
    pre_transform=train_val_test_pre_transforms
)
val_test_dataset = LOL_Dataset(
    root=ROOT_DIR,
    transform=val_test_transformer,
    pre_transform=train_val_test_pre_transforms
)
# %%
LOL_GRAPH = train_dataset[0]
DATA_SET_NUM_FEATURES = LOL_GRAPH.num_node_features
NUM_NODES = LOL_GRAPH.num_nodes
DATA_SET_CLASSES = train_dataset.num_classes
# LOL_GRAPH = train_dataset[0]
# %%
print(f"{DATA_SET_NUM_FEATURES = }")
print(f"{DATA_SET_CLASSES = }")
print(f"{NUM_NODES = }")
print(f"{LOL_GRAPH.num_edges = }")
print(f'Has isolated nodes: {LOL_GRAPH.has_isolated_nodes()}')
# %%
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_test_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(val_test_dataset, batch_size=BATCH_SIZE)
# %%[markdown]
# # Building model layers
# %%
gnn_layers: list[MessagePassing] = [
    GCNConv(DATA_SET_NUM_FEATURES, HIDDEN_CHANNELS),
    GCNConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS),
    GCNConv(HIDDEN_CHANNELS, DATA_SET_CLASSES)
]

activation_functions: list[Callable] = [
    F.tanh,
    F.tanh,
    F.tanh
]
print(f"{gnn_layers = }")
print(f"{activation_functions = }")
# %%
gcn = GNNModel(gnn_layers, activation_functions, addDropOut=False)
gcn_optimizer = torch.optim.Adam(gcn.parameters(), lr=0.001)
gcn_criterion = torch.nn.CrossEntropyLoss()
print(gcn)
LOL_GRAPH.to(torch.device("cpu"))
print(summary(gcn, LOL_GRAPH.x, LOL_GRAPH.edge_index))
# %%
# %%[markdown]
#  ## Visualize **before** training
gcn.eval()
# LOL_GRAPH.to(torch.device("cpu"))
vis_model_out, vis_hidden_embeddings = gcn(LOL_GRAPH.x, LOL_GRAPH.edge_index)
print(f'Embedding shape: {list(vis_hidden_embeddings.shape)}')
visualize(vis_hidden_embeddings, color=LOL_GRAPH.y)
visualize2(vis_model_out, color=LOL_GRAPH.y)
# %%
trainPhase(
    train_loader=train_loader,
    val_loader=val_loader,
    num_classes=DATA_SET_CLASSES,
    epochs=NUM_EPOCHS,
    model=gcn,
    criterion=gcn_criterion,
    optimizer=gcn_optimizer,
    device=DEVICE,
    save_experiment=True,
    experiment_output_dir=OUTPUT_DIR
)

# %%[markdown]
#  # **Results**

#  #### Model Test
# %%
test_loss, test_accuracy = testPhase(gcn, gcn_criterion, test_loader, DEVICE)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {100 * test_accuracy:.2f}%')

# %%
# %%[markdown]
#  ## Visualize Results after **training**
# %%
gcn.eval()
LOL_GRAPH.to(torch.device("cuda"))
vis_model_out2, vis_hidden_embeddings2 = gcn(LOL_GRAPH.x, LOL_GRAPH.edge_index)
vis_hidden_embeddings2 = vis_hidden_embeddings2.to(torch.device("cpu"))
vis_model_out2 = vis_model_out2.to(torch.device("cpu"))
print(f'Embedding shape: {list(vis_hidden_embeddings2.shape)}')
visualize(vis_hidden_embeddings2, color=LOL_GRAPH.y)

# %%
visualize2(vis_model_out2, color=LOL_GRAPH.y)

# %%
def save_output_graph(hidden_embeddings: torch.Tensor, edge_index: torch.Tensor, labels: torch.Tensor):
    """
    
    detaches hidden_embeddings and labels to CPU might need to them back to original device if needed
    """
    graph_ptg = Data(x=hidden_embeddings, edge_index=edge_index, y=labels)
    graph_nx = to_networkx(graph_ptg)
    for node, node_feature in enumerate(hidden_embeddings):
        node_feature = node_feature.detach().cpu().numpy()
        graph_nx.nodes[node].update({feature: float(value) for feature, value in enumerate(node_feature)})
    # graph_nx.graph["y"] = 
    labels = labels.detach().cpu().numpy()
    graph_nx.graph.update({f"node{node}_label": label for node, label in enumerate(labels)})
    graph_nx.graph
    date_time_str = get_now_datetime()
    GRAPHML_FILENAME = f"hidden_embedding_graph{date_time_str}"
    nx.write_graphml(graph_nx, f"{GRAPHML_FILENAME}.graphml")

# %%
save_output_graph(vis_hidden_embeddings2, LOL_GRAPH.edge_index, LOL_GRAPH.y)
# %%
