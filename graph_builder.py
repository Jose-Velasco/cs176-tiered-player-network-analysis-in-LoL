# %%
import pandas as pd
import networkx as nx
from utils import get_now_datetime
import ast
# %%
DATASET_PATH = "lol_C_M_GM_processed_normalized_D04-21-2024T20_04_54_with_lp.csv"
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
GRAPHML_FILENAME = f"lol_C_M_GM_graph_{date_time_str}_with_lp"
nx.write_graphml(G, f"{GRAPHML_FILENAME}.graphml")

# %%
