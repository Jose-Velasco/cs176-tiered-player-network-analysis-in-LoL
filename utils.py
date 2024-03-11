from bs4 import BeautifulSoup, NavigableString, Tag
from enums import Tier, Position
from data_classes import Summoner
import re
import json
from collections import Counter
from proxy_utils import ProxyRoller

def extract_summoners(table: Tag | NavigableString)-> list[Summoner]:
    """
    helper function to extract all players in a given table
    """
    rows = table.find_all('tr')
    summoners: list[Summoner] = []
    for row in rows[1:]:
        username = row.find("span", class_=["css-ao94tw", "e1swkqyq1"]).text
        tagline = row.find("span", class_=["css-1mbuqon", "e1swkqyq2"]).text
        tier_raw = row.find("td", class_=["css-13jn5d5", "euud7vz3"]).text
        tier = Tier[tier_raw.upper()]
        lp_raw: str = row.find("td", class_=["css-1oruqdu", "euud7vz4"]).find("span").text
        lp_raw_cleaned = re.sub(r"[,\s]+|LP", "", lp_raw)
        lp = int(lp_raw_cleaned)
        level_raw = row.find("td", class_=["css-139lfew", "euud7vz5"]).text
        level = int(level_raw)
        total_wins_raw = row.find("div", class_="w").text
        total_wins_cleaned = re.sub("W", "", total_wins_raw)
        total_wins = int(total_wins_cleaned)
        total_losses_raw = row.find("div", class_="l").text
        total_losses_cleaned = re.sub("L", "", total_losses_raw)
        total_losses = int(total_losses_cleaned)

        summoner = Summoner(
            username=username,
            tagline=tagline,
            tier=tier,
            lp=lp,
            level=level,
            total_wins=total_wins,
            total_losses=total_losses
        )
        summoners.append(summoner)
    return summoners

#  testing still
def get_all_summoners(tiers: list[Tier], base_url: str, leaderboard_url: str, proxy_roller: ProxyRoller) -> list[Summoner]:
    """
    gets all summoners in the provided tiers from the leaderboard_rul
    """
    summoners: list[Summoner] = []
    for tier in tiers:
        pg = 1
        table_found = True
        while (table_found):
            lboard_page_url = f"{base_url}{leaderboard_url}{tier.value}&page={pg}"
            res = proxy_roller.get(lboard_page_url)
            soup = BeautifulSoup(res.text, "lxml")
            table = soup.find("table", class_=["css-1l95r9q", "euud7vz10"])
            if table:
                summoners += extract_summoners(table)
                pg += 1
            else:
                table_found = False
    return summoners

def get_summoner_id(user_name: str, tagline: str, base_url: str, summoner_detail_url: str, proxy_roller: ProxyRoller) -> str:
    """
    gets a summoners_id based on username and tagline
    space = %20
    # = -
    """
    summoner = f"{user_name.replace(" ", "%20")}{tagline.replace("#", "-")}"
    url =  f"{base_url}{summoner_detail_url}{summoner}"
    res = proxy_roller.get(url)
    soup = BeautifulSoup(res.text, "lxml")
    script_raw = soup.find("script", id="__NEXT_DATA__").text
    script_json = json.loads(script_raw)
    return script_json["props"]["pageProps"]["data"]["summoner_id"]

def get_summoner_details_past_n_games(summoner_id: str, proxy_roller: ProxyRoller, n: int = 20) -> dict[str, int | float | str]:
    """
    send request to api to get player details
    """
    api_url = f"https://op.gg/api/v1.0/internal/bypass/games/na/summoners/{summoner_id}?&limit={n}&hl=en_US&game_type=soloranked"
    api_summer_detail_res = proxy_roller.get(api_url)

    games = 0
    wins = 0
    losses = 0
    kills = 0
    deaths = 0
    assists = 0
    positions_played = Counter()
    minion_kill = 0
    neutral_minion_kill = 0

    for game in api_summer_detail_res.json()["data"]:
        games += 1
        wins += 1 if game["myData"]["stats"]["result"] == "WIN" else 0
        losses += 1 if game["myData"]["stats"]["result"] == "LOSE" else 0
        kills += int(game["myData"]["stats"]["kill"])
        deaths += int(game["myData"]["stats"]["death"])
        assists += int(game["myData"]["stats"]["assist"])
        minion_kill += int(game["myData"]["stats"]["minion_kill"])
        neutral_minion_kill += int(game["myData"]["stats"]["neutral_minion_kill"])
        positions_played.update([game["myData"]["position"]])

    cs_avg = (minion_kill + neutral_minion_kill) / games
    kills_avg = kills / games
    deaths_avg = deaths / games
    assists_avg = assists / games
    preferred_position = Position[positions_played.most_common(1)[0][0].upper()]
    return {
        "games_found": games,
        "wins": wins,
        "losses": losses,
        "kills_avg": kills_avg,
        "deaths_avg": deaths_avg,
        "assists_avg": assists_avg,
        "cs_avg": cs_avg,
        "preferred_position": preferred_position
    }

def update_summoner(summoner: Summoner, data: dict[str, str]) -> None:
    """
    helper function to update some properties
    """
    summoner.past_n_wins = data["wins"]
    summoner.past_n_losses = data["losses"]
    summoner.kills = data["kills_avg"]
    summoner.deaths = data["deaths_avg"]
    summoner.assists = data["assists_avg"]
    summoner.creep_score = data["cs_avg"]
    summoner.preferred_position = data["preferred_position"]