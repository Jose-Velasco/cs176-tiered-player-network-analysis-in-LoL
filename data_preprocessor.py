# %%
import pandas as pd
from utils import one_hot_encode, get_now_datetime
# %%
DATASET_PATH = "./lol_C_M_GM_raw_unnormalized_D03-31-2024T00_47_53.csv"
DATA_TYPES = {
    "username": "string",
    "tagline": "string",
    "summoner_id": "string",
    "tier": "category",
    "preferred_position": "category"
}
# %%
lol = pd.read_csv(DATASET_PATH, dtype=DATA_TYPES)
# %%
lol.info()
# %%
# %%[markdown]
# ## One hot encode categorical data
# %%
lol = one_hot_encode(lol, "preferred_position", concat=True)
# %%
lol = one_hot_encode(lol, "tier", concat=True)
# %%
# %%[markdown]
# ## normalizing int/float columns by number of ranked games found
# creep_score, damage_dealt_to_champions, damage_taken, control_wards_placed, wards_placed, wards_kills
# kills, deaths, assists
# %%
lol["creep_score"] = lol["creep_score"] / lol["games_found"]
lol["damage_dealt_to_champions"] = lol["damage_dealt_to_champions"] / lol["games_found"]
lol["damage_taken"] = lol["damage_taken"] / lol["games_found"]
lol["control_wards_placed"] = lol["control_wards_placed"] / lol["games_found"]
lol["wards_placed"] = lol["wards_placed"] / lol["games_found"]
lol["wards_kills"] = lol["wards_kills"] / lol["games_found"]
lol["kills"] = lol["kills"] / lol["games_found"]
lol["deaths"] = lol["deaths"] / lol["games_found"]
lol["assists"] = lol["assists"] / lol["games_found"]

# %%[markdown]
# ## removing not needed columns
# %%
lol.drop("preferred_position", axis=1, inplace=True)
lol.drop("tier", axis=1, inplace=True)
lol.drop("summoner_id", axis=1, inplace=True)
lol.drop("lp", axis=1, inplace=True)
lol.drop("games_found", axis=1, inplace=True)
# %%
# %%[markdown]
# ## saving processed dataset
# %%
date_time_str = get_now_datetime()
CSV_FILENAME = f"lol_C_M_GM_processed_normalized_{date_time_str}.csv"
lol.to_csv(CSV_FILENAME, index=False)