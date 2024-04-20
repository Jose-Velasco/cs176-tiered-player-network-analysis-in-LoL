# %%
import pandas as pd
import numpy as np
import networkx as nx
from utils import get_now_datetime
import ast
import torch
# %%
# DATASET_PATH = "lol_C_M_GM_processed_normalized_D03-31-2024T22_12_00.csv"
DATASET_PATH = "lol_C_M_GM_processed_normalized_D04-19-2024T12_33_31.csv"
DATA_TYPES = {
    "username": "string",
    "tagline": "string",
}
# %%
lol = pd.read_csv(DATASET_PATH, dtype=DATA_TYPES)
lol.info()
# %%
nodes = lol["username"] + "#" + lol["tagline"]
nodes.head(15)
# %%
G = nx.Graph()
# %%
G.add_nodes_from(nodes)
# %%[markdown]
# ## create Graph for use in virilization tools like Gephi
# %%
for idx, row in lol.iterrows():
    current_player = row["username"] + "#" + row["tagline"]

    node_features = row.drop(labels=["username", "tagline", "players_played_with"]).to_dict()
    G.nodes[current_player].update(node_features)

    # convert column players_played_with string list as a python list data type
    players_played_with = ast.literal_eval(row["players_played_with"])
    for participant in players_played_with:
        if G.has_node(participant):
            G.add_edge(current_player, participant)
# %%
date_time_str = get_now_datetime()
GRAPHML_FILENAME = f"lol_C_M_GM_graph_{date_time_str}_A"
nx.write_graphml(G, f"{GRAPHML_FILENAME}.graphml")

# %%[markdown]
# ## create Graph for use in downstream graph nerual network task (no longer needed)
# %%
# G.graph["x_labels"] = lol.iloc[0].index.to_list()
# for idx, row in lol.iterrows():
#     if G.graph["x_labels"] != row.index.to_list():
#         print("Warning: node feature vector elements are out of order, there will be issue in downstream task!!!")

#     current_player = row["username"] + "#" + row["tagline"]
#     all_node_features_dict: dict = G.nodes[current_player].copy()
#     G.nodes[current_player].clear()
#     G.nodes[current_player]["x"] = torch.tensor(np.array(all_node_features_dict.values()), dtype=torch.float32)

# PYTORCH_FILENAME = f"lol_C_M_GM_graph_{date_time_str}_B.pt"
# torch.save(G, PYTORCH_FILENAME)
# %%
# TESTING
# G.graph["x_labels"]
# %%
row.index.to_list()
# %%
G.nodes["yuu13#sus"]
# %%
