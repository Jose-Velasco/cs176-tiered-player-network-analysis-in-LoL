# %%
import torch
from pytorch_geometric_dataset import LOL_Dataset
from torch_geometric.transforms import NormalizeFeatures, Compose, ToDevice
from train_utils import testPhase, trainPhase, CustomRandomNodeSplitMasker, CustomRandomNodeUnderSampler,ConcatNodeCentralities



# %%
ROOT_DIR="PTG_data"
TEST_SIZE = 0.1
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%

# nodeUnderSample = CustomRandomNodeUnderSampler(RANDOM_STATE)
# nodeSPlitMasker = CustomRandomNodeSplitMasker(
#     test_size=TEST_SIZE,
#     validation_size=VALIDATION_SIZE,
#     random_state=RANDOM_STATE
# )
train_transformer = Compose(
    [
        NormalizeFeatures(),
        # ToDevice(DEVICE),
        # nodeUnderSample,
        # nodeSPlitMasker,
        # ConcatNodeCentralities()
    ]
)
# %%

train_dataset = LOL_Dataset(
    root=ROOT_DIR,
    transform=train_transformer,
    pre_transform=ConcatNodeCentralities()
)

# %%
train_dataset[0]
# %%
train_dataset[0].num_nodes
train_dataset[0].num_node_features

# %%
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import networkx as nx
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch_geometric.data import Data
import numpy as np
# %%
def plot_graph(graph, title: str, figSize: tuple[int, int], nodeSize: int, edgeWidth: int):
    if isinstance(graph, Data):
        graph_nx = to_networkx(graph, node_attrs=["x"], edge_attrs=["edge_attr"], graph_attrs=["y"])
        node_features_dict = nx.get_node_attributes(graph_nx, "x")
        node_features = [node_features_dict[index] for index in range(len(node_features_dict))]
        node_features_df = pd.DataFrame(node_features)
        node_features_means = node_features_df.mean(axis=0)
    else:
        graph_nx = graph
        node_features_means_list: list = []
        for node in graph.nodes:
            node_feature: list = graph.nodes[node].values()
            node_features_means_list.append(sum(node_feature) / len(node_feature))
        node_features_means = np.array(node_features_means_list)

    plt.style.use('default')
    plt.rcParams['axes.facecolor'] = 'white'
    fig = plt.figure(figsize=figSize)
    pos = nx.nx_agraph.graphviz_layout(graph_nx, prog="sfdp")
    # Draw the edges and the nodes with features
    nx.draw_networkx_edges(
        graph_nx,
        pos,
        width=edgeWidth,
        edge_color='black',
        # arrowsize=15,
        # min_source_margin= 14,
        # min_target_margin=14,
    )

    nx.draw_networkx_nodes(
        graph_nx, pos,
        node_size=nodeSize,
        node_color=node_features_means,
        cmap="cool",
    )

    # nx.draw_networkx_nodes(
    #     graph_nx,
    #     pos,
    #     node_color=color,
    #     cmap="cool",
    #     alpha=0.5
    # )

    # Add a colorbar for the node features
    sm = plt.cm.ScalarMappable(cmap="cool", norm=plt.Normalize(vmin=node_features_means.min(), vmax=node_features_means.max()))
    sm._A = []
    # [left, bottom, width, height]
    cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    plt.colorbar(sm, cax=cax)
    plt.title(title)
    plt.axis('off')
    plt.show()

# %%
graph_model_output_fileName = "hidden_embedding_graphD04-23-2024T19_36_57.graphml"
graph_model_output_nx: nx.Graph = nx.read_graphml(graph_model_output_fileName)
# %%
nx.is_weakly_connected(graph_model_output_nx)
# %%
plot_graph(
    graph=graph_model_output_nx,
    title="Graph showing hidden embeddings out of a trained GCN model",
    figSize=(20, 20),
    nodeSize=800,
    edgeWidth=1
)
# %%
