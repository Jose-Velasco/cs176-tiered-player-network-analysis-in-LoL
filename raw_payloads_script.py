# %%
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from enums import Tier
import time
from utils import generate_all_summoners_payload
from datetime import datetime
from dateutil import tz
import json

# %%[markdown]
# ## Script uses undetected_chromedriver (selenium) will download full raw json payloads for players in TIERS_EXTRACTING adn their details

#  %%
# PRE_LEADERBOARD_URL = "https://www.op.gg/_next/data/_VrQ9gbyiIZBzMmfdcsiX/en_US/leaderboards/tier.json?tier=challenger&page=1"
PRE_LEADERBOARD_URL = "https://www.op.gg/_next/data/NvO79ew7NNfVMLtqz7wzX/en_US/leaderboards/tier.json?tier={}&page={}"
SUMMONER_PROFILE_DETAILS_URL = "https://op.gg/api/v1.0/internal/bypass/games/na/summoners/{}?&limit={}&hl=en_US&game_type=soloranked"
TIERS_EXTRACTING = [Tier.CHALLENGER, Tier.GRANDMASTER, Tier.MASTER]
pst = tz.gettz('America/Los_Angeles')
# %%
driver = uc.Chrome()

# %%
summoner_ids: list[str] = []
for idx, payload in enumerate(generate_all_summoners_payload(TIERS_EXTRACTING, PRE_LEADERBOARD_URL, driver)):
    now_datetime = datetime.now(pst)
    # D=date, T=time
    date_time_str = now_datetime.strftime("D%m-%d-%YT%H_%M_%S")
    file_path = f"./leader_board_payloads/lead_board_payload{idx}_{date_time_str}.json"
    json_str = json.dumps(payload, indent=4)
    for summoner_data in payload["pageProps"]["data"]:
        summoner_ids.append(summoner_data["summoner"]["summoner_id"])

    with open(file_path, "w") as json_file:
        json_file.write(json_str)
# %%
num_games: int = 20
for index, id in enumerate(summoner_ids):
    if index % 20 == 0:
        print(f"{index = }")
        time.sleep(1)
    api_url = SUMMONER_PROFILE_DETAILS_URL.format(id, num_games)
    driver.get(api_url) 
    soup = BeautifulSoup(driver.page_source, "lxml")
    raw_json = soup.find("body").text

    now_datetime = datetime.now(pst)
    # D=date, T=time
    date_time_str = now_datetime.strftime("D%m-%d-%YT%H_%M_%S")
    file_path = f"./player_detail_payloads/player_{id}_{date_time_str}.json"
    with open(file_path, "w") as json_file:
        json_file.write(raw_json)


# %%
print("DONE!!!!!!!")
