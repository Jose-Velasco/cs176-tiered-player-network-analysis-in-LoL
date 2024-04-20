#  %%
import pandas as pd
from utils import extract_summoners_payload, update_summoner_details_and_players_played_with
from pathlib import Path
import json
from data_classes import Summoner
from datetime import datetime
from dateutil import tz

#  %%
LEADERBOARD_PATH = "./leader_board_payloads"
PLAYERS_DETAIL_PATH = "./player_detail_payloads"
NORMALIZE_STATUS = False
#  %%
# load all summoners
leader_board_directory = Path(LEADERBOARD_PATH)
summoners: list[Summoner] = []
for leaderboard_file_path in leader_board_directory.iterdir():
    if leaderboard_file_path.is_file():
        with open(leaderboard_file_path, "r") as file:
            data = json.load(file)
            summoners += extract_summoners_payload(data)
# %%
# create dict to be able to find summoners based on their username and tagline quickly
# data in leaderboard only has partial summoner info and the rest is in players details
summoners_with_details_dict: dict[str, Summoner] = {}
for idx, summoner in enumerate(summoners):
    summoner_key = f"{summoner.summoner_id}"
    if summoner_key in summoners_with_details_dict:
        print(f"Warning: duplicate summoner at index = {idx} found \n{summoner.username = }\n{summoner.tagline = }\n{summoner.summoner_id}\nsummoner not added\n")
    else:
        summoners_with_details_dict[summoner_key] = summoner
# %%
# dup_ids = [
#     "hA52229gWLzXEfTyB6wU9zORV7GE19UhJXbdAEbUaGeaAa0",
#     "RNqzRc9koSthRn8QaxfE752ykvsYIJoVFtuJ22hY1mONDO9_WH6P_on6Eg",
#     "65tVObN0l_QxmX3u8a2T4TMs_62r5nOc3mWIyfWg2M42MeM",
#     "VkFB0qfPk2UcYqyJx_JnJLjxF84QjjVbjpzH_81qUOIGf90",
#     "ZepVzz1jq0_7Lgf2c34WnxvU7uSWmE2D-aMOYMTXoz32tXw",
#     "zMyNGydhn1j-CRpPaV0-Z5NQ8NDGjq8y-ULpcsj-h_A5zPRErcnOWYSQ0Q"
# ]
# for summoner in summoners:
#     if summoner.summoner_id in dup_ids:
#         print(summoner)

#  %%
# getting player details from payloads
players_detail_directory = Path(PLAYERS_DETAIL_PATH)
for player_detail_file_path in players_detail_directory.iterdir():
    if not player_detail_file_path.is_file():
        continue
    with open(player_detail_file_path, "r") as file:
        data = json.load(file)
        # payload_owner_key: str = f"{data['data'][0]['myData']['summoner']['game_name']}{data['data'][0]['myData']['summoner']['tagline']}"
        payload_owner_key: str = f"{data['data'][0]['myData']['summoner']['summoner_id']}"
        summoner: Summoner = summoners_with_details_dict[payload_owner_key]
        update_summoner_details_and_players_played_with(summoner, data, NORMALIZE_STATUS)

# %%[markdown]
# ## Saving dataset to CSV file 
# %%
lol_df = pd.DataFrame(list(summoners_with_details_dict.values()))
# %%
pst = tz.gettz('America/Los_Angeles')

now_datetime = datetime.now(pst)
# D=date, T=time
date_time_str = now_datetime.strftime("D%m-%d-%YT%H_%M_%S")

norm_status: str = "normalized" if NORMALIZE_STATUS else "unnormalized"
CSV_FILENAME = f"lol_C_M_GM_raw_{norm_status}_{date_time_str}.csv"

lol_df.to_csv(CSV_FILENAME, index=False)
# %%
