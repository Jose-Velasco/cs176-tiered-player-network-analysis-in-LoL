# %%
from typing import Any, Callable
import torch
from torch_geometric.data import Dataset, Data
import os.path as osp
from pathlib import Path
import networkx as nx
from torch_geometric.utils.convert import from_networkx
import numpy as np

# %%
class LOL_Dataset(Dataset):
    def __init__(self, root: str | None = None, transform: Callable[..., Any] | None = None, pre_transform: Callable[..., Any] | None = None, pre_filter: Callable[..., Any] | None = None, log: bool = True, force_reload: bool = False) -> None:
        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)
    
    @property
    def raw_file_names(self):
        directory = Path(f"{self.root}/raw")
        # might need to be changed yo file paths instead of just names if errors with processed_file_names(...)
        file_names = [file_path.name for file_path in directory.glob('*') if file_path.is_file()]
        return file_names

    @property
    def processed_file_names(self):
        rawFiles: list[str] = self.raw_file_names
        processedFileNames: list[str] = []
        for idx, file in enumerate(rawFiles):
            filepath = f"data_{idx}.pt"
            processedFileNames.append(filepath)
        return processedFileNames

    def download(self):
        """
        our data is stored locally
        Download to `self.raw_dir`.
        """
        pass

    def process(self):
        # creates a single graph for each dataset
        for idx, raw_filename in enumerate(self.raw_file_names):
            # shape [num_nodes, num_node_features]
            G = nx.read_graphml(f"{self.raw_dir}/{raw_filename}")
            output_graph_fileName = f"data_{idx}.pt"
            # first_node, first_node_feat = next(G.nodes(data=True))
            node_attrs_names = list(next(iter(G.nodes(data=True)))[-1].keys())
            # tier holds the label/targets of the node
            if 'tier' in node_attrs_names:
                node_attrs_names.remove("tier")
            from_nx_graph: Data = from_networkx(G, group_node_attrs=node_attrs_names)
            #  shape [num_nodes, *]
            node_level_targets: list[int] = [node_attr_dict.get('tier', None) for _, node_attr_dict in G.nodes(data=True)]
            if None in node_level_targets:
                raise ValueError("Some nodes are missing the 'tier' attribute.")
            else:
                # Convert targets to torch.Tensor and add them to the Data object
                from_nx_graph.y = torch.tensor(np.asarray(node_level_targets), dtype=torch.int64)

            save_graph = Data(
                x=torch.tensor(from_nx_graph.x, dtype=torch.float32),
                edge_index=from_nx_graph.edge_index,
                y=from_nx_graph.y
            )

            if self.pre_filter is not None and not self.pre_filter(save_graph):
                continue

            if self.pre_transform is not None:
                save_graph = self.pre_transform(save_graph)

            torch.save(save_graph, osp.join(self.processed_dir, output_graph_fileName))


    def len(self):
        # TODO speed up?
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data