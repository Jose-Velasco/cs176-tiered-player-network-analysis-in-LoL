import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import networkx as nx
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch_geometric.data import Data

def plot_graph(graph, title: str, figSize: tuple[int, int], nodeSize: int, edgeWidth: int):
    if isinstance(graph, Data):
        graph_nx = to_networkx(graph, node_attrs=["x"], edge_attrs=["edge_attr"], graph_attrs=["y"])
    else:
        graph_nx = graph

    node_features_dict = nx.get_node_attributes(graph_nx, "x")
    node_features = [node_features_dict[index] for index in range(len(node_features_dict))]
    node_features_df = pd.DataFrame(node_features)

    plt.style.use('default')
    plt.rcParams['axes.facecolor'] = 'white'
    fig = plt.figure(figsize=figSize)

    pos = nx.nx_agraph.graphviz_layout(graph_nx)
    node_features_means = node_features_df.mean(axis=0)

    # Draw the edges and the nodes with features
    # nx.draw_networkx_edges(
    #     graph_nx,
    #     pos,
    #     width=edgeWidth,
    #     edge_color='black',
    #     arrowsize=15,
    #     min_source_margin= 14,
    #     min_target_margin=14,
    #     with_labels= False
    # )

    # nx.draw_networkx_nodes(
    #     graph_nx, pos,
    #     node_size=nodeSize,
    #     node_color=node_features_means,
    #     cmap="cool",
    #     with_labels= False
    # )
    
    pos = nx.spring_layout(graph_nx, seed=42)
    nx.draw_networkx(
        graph_nx,
        pos=pos,
        with_labels=False,
        cmap="Set2"
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
    cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    plt.colorbar(sm, cax=cax)
    plt.title(title)
    plt.axis('off')
    plt.show()


def visualize(h, color, epoch=None, loss=None, accuracy=None):
    plt.figure(figsize=(12,12))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        color = color.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=70, c=color, cmap="Set2", alpha=0.5)
        if epoch is not None and loss is not None and accuracy['train'] is not None and accuracy['val'] is not None:
            plt.xlabel((f'Epoch: {epoch}, Loss: {loss.item():.4f} \n'
                        f'Training Accuracy: {accuracy["train"]*100:.2f}% \n'
                        f' Validation Accuracy: {accuracy["val"]*100:.2f}%'),
                        fontsize=16)
    else:
        pos = nx.spring_layout(h, seed=42)
        # nx.draw_networkx(h, pos=pos, with_labels=False, cmap="Set2")
        nx.draw_networkx_nodes(
            h,
            pos,
            node_color=color,
            cmap="cool",
            alpha=0.5
        )

    plt.show()

def visualize2(h, color):
    h = h.detach().cpu().numpy()
    color = color.detach().cpu().numpy()
    z = TSNE(n_components=2).fit_transform(h)

    plt.figure(figsize=(12,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2", alpha=0.5)
    plt.show()