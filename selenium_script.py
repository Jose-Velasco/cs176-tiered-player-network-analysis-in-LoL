# %%
import undetected_chromedriver as uc
from enums import Tier
import time
import pandas as pd
from utils import get_all_summoners_payload, get_summoner_profile_details_past_n_games
from datetime import datetime
from dateutil import tz

# %%[markdown]
# ## Script uses undetected_chromedriver (selenium) for downloading players within TIERS_EXTRACTING and extracting specific details/stats of a player

#  %%
# BASE_URL = "https://www.op.gg/"
PRE_LEADERBOARD_URL = "https://www.op.gg/_next/data/_VrQ9gbyiIZBzMmfdcsiX/en_US/leaderboards/tier.json?tier={}&page={}"
# SUMMONER_DETAIL_URL = "summoners/na/"
SUMMONER_PROFILE_DETAILS_URL = "https://op.gg/api/v1.0/internal/bypass/games/na/summoners/{}?&limit={}&hl=en_US&game_type=soloranked"
TIERS_EXTRACTING = [Tier.CHALLENGER, Tier.GRANDMASTER, Tier.MASTER]
# %%
# driver = uc.Chrome()

# %%
summoners = get_all_summoners_payload(TIERS_EXTRACTING, PRE_LEADERBOARD_URL, driver)
# %%
for idx, summoner in enumerate(summoners[141:]):
    if idx % 10 == 0:
        print(f"{idx = }")
        time.sleep(1)
    get_summoner_profile_details_past_n_games(SUMMONER_PROFILE_DETAILS_URL, summoner, driver)

# %%
lol_df = pd.DataFrame(summoners)
# %%
pst = tz.gettz('America/Los_Angeles')

now_datetime = datetime.now(pst)
# D=date, T=time
date_time_str = now_datetime.strftime("D%m-%d-%YT%H_%M_%S")

CSV_FILENAME = f"lol_C_M_GM{date_time_str}.csv"

lol_df.to_csv(CSV_FILENAME, index=False)