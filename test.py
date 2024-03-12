# %%
import requests
from bs4 import BeautifulSoup
from enums import Tier
from data_classes import Summoner
from utils import extract_summoners, get_summoner_profile_details_past_n_games, get_summoner_id
from proxy_utils import get_valid_proxies, ProxyRoller

# %%
BASE_URL = "https://www.op.gg/"
LEADERBOARDS_URL = "leaderboards/tier?tier="
SUMMONER_DETAIL_URL = "summoners/na/"
SUMMONER_PROFILE_DETAILS_URL = "https://op.gg/api/v1.0/internal/bypass/games/na/summoners/{}?&limit={}&hl=en_US&game_type=soloranked"
TIERS_EXTRACTING = [Tier.CHALLENGER, Tier.GRANDMASTER, Tier.MASTER]
# %%
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'}
# %%
# get players from first page that are challenger
res = requests.get("https://www.op.gg/leaderboards/tier?tier=challenger&page=1", headers=headers)
#  %%
soup = BeautifulSoup(res.text, "lxml")

# %%
# get the table on th page
table = soup.find("table", class_=["css-1l95r9q", "euud7vz10"])

# %%
summoners: list[Summoner] = extract_summoners(table)

# %%
summoner0 = summoners[0]
summoner0.summoner_id = get_summoner_id(summoner0.username, summoner0.tagline, BASE_URL, SUMMONER_DETAIL_URL, headers)

# %%
# print(summoner0)

# TODO: test get_valid_proxies, ProxyRoller, get_all_summoners, get_summoner_profile_details_past_n_games
# # %%
# valid_proxies = get_valid_proxies("all_proxy_list.txt")

# # %%
# proxy_roller = ProxyRoller(
#     session=requests.Session(),
#     proxies=valid_proxies,
#     headers=headers
# )

# # %%
# all_summoners = get_all_summoners(TIERS_EXTRACTING, BASE_URL, LEADERBOARDS_URL, proxy_roller)

#  %%
# send request to api to get player details and players they played with last 20 solo/duo ranked games
# summoner_details = get_summoner_profile_details_past_n_games(SUMMONER_PROFILE_DETAILS_URL, summoner0, proxy_roller)
