from typing import Any
from bs4 import BeautifulSoup, NavigableString, Tag
from enums import Tier, Position
from data_classes import Summoner
import re
import json
from collections import Counter
from http_clients import HttpClient
import time


def extract_payload_as_json(driver) -> dict:
    soup = BeautifulSoup(driver.page_source, "lxml")
    dict_from_json = json.loads(soup.find("body").text)
    return dict_from_json
# def extract_summoners(table: Tag | NavigableString)-> list[Summoner]:
#     """
#     helper function to extract all players in a given table
#     """
#     rows = table.find_all('tr')
#     summoners: list[Summoner] = []
#     for row in rows[1:]:
#         username = row.find("span", class_=["css-ao94tw", "e1swkqyq1"]).text
#         tagline = row.find("span", class_=["css-1mbuqon", "e1swkqyq2"]).text
#         tier_raw = row.find("td", class_=["css-13jn5d5", "euud7vz3"]).text
#         tier = Tier[tier_raw.upper()]
#         lp_raw: str = row.find("td", class_=["css-1oruqdu", "euud7vz4"]).find("span").text
#         lp_raw_cleaned = re.sub(r"[,\s]+|LP", "", lp_raw)
#         lp = int(lp_raw_cleaned)
#         level_raw = row.find("td", class_=["css-139lfew", "euud7vz5"]).text
#         level = int(level_raw)
#         total_wins_raw = row.find("div", class_="w").text
#         total_wins_cleaned = re.sub("W", "", total_wins_raw)
#         total_wins = int(total_wins_cleaned)
#         total_losses_raw = row.find("div", class_="l").text
#         total_losses_cleaned = re.sub("L", "", total_losses_raw)
#         total_losses = int(total_losses_cleaned)

#         summoner = Summoner(
#             username=username,
#             tagline=tagline,
#             tier=tier,
#             lp=lp,
#             level=level,
#             total_wins=total_wins,
#             total_losses=total_losses
#         )
#         summoners.append(summoner)
#     return summoners

def extract_summoners_payload(payload: dict[str, Any])-> list[Summoner]:
    """
    helper function to extract all players in a given table
    """
    summoner_payloads = payload["pageProps"]["data"]
    summoners: list[Summoner] = []
    for summoner_payload in summoner_payloads:
        summoner_data = summoner_payload["summoner"]
        username = summoner_data["game_name"]
        tagline = summoner_data["tagline"]
        summoner_id = summoner_data["summoner_id"]
        tier_raw = summoner_data["league_stats"][0]["tier_info"]["tier"]
        tier = Tier[tier_raw.upper()]
        lp = int(summoner_data["league_stats"][0]["tier_info"]["lp"])
        level = int(summoner_data["level"])
        total_wins = int(summoner_data["league_stats"][0]["win"])
        total_losses = int(summoner_data["league_stats"][0]["lose"])

        summoner = Summoner(
            username=username,
            tagline=tagline,
            summoner_id=summoner_id,
            tier=tier,
            lp=lp,
            level=level,
            total_wins=total_wins,
            total_losses=total_losses
        )
        summoners.append(summoner)
    return summoners


#  testing still
# def get_all_summoners(tiers: list[Tier], base_url: str, leaderboard_url: str, http_client: HttpClient) -> list[Summoner]:
#     """
#     gets all summoners in the provided tiers from the leaderboard_rul
#     """
#     summoners: list[Summoner] = []
#     for tier in tiers:
#         pg = 1
#         table_found = True
#         while (table_found):
#             lboard_page_url = f"{base_url}{leaderboard_url}{tier.value}&page={pg}"
#             res = http_client.get(lboard_page_url)
#             soup = BeautifulSoup(res.text, "lxml")
#             table = soup.find("table", class_=["css-1l95r9q", "euud7vz10"])
#             if table:
#                 summoners += extract_summoners(table)
#                 pg += 1
#             else:
#                 table_found = False
#     return summoners

def get_all_summoners_payload(tiers: list[Tier], pre_leaderboard_url: str, http_client: HttpClient, delay: int = 1, batch: int = 10) -> list[Summoner]:
    """
    gets all summoners in the provided tiers from the leaderboard_rul
    """
    summoners: list[Summoner] = []
    for tier in tiers:
        pg = 1
        table_found = True
        while (table_found):
            if pg % batch == 0:
                time.sleep(delay)
            
            # lboard_page_url = f"{base_url}{leaderboard_url}{tier.value}&page={pg}"
            lboard_page_url = pre_leaderboard_url.format(tier, pg)
            http_client.get(lboard_page_url)
            res_json = extract_payload_as_json(http_client)
            if len(res_json["pageProps"]["data"]) != 0:
                summoners += extract_summoners_payload(res_json)
                pg += 1
            else:
                table_found = False
    return summoners

def generate_all_summoners_payload(tiers: list[Tier], pre_leaderboard_url: str, http_client: HttpClient, delay: int = 1, batch: int = 10):
    """
    generator for all summoners in the provided tiers from the leaderboard_rul as JSON
    """
    for tier in tiers:
        pg = 1
        data_found = True
        while (data_found):
            if pg % batch == 0:
                time.sleep(delay)
            
            lboard_page_url = pre_leaderboard_url.format(tier, pg)
            http_client.get(lboard_page_url)
            res_json = extract_payload_as_json(http_client)
            if len(res_json["pageProps"]["data"]) != 0:
                yield res_json
                pg += 1
            else:
                data_found = False

def get_summoner_id(user_name: str, tagline: str, base_url: str, summoner_detail_url: str, http_client: HttpClient) -> str:
    """
    gets a summoners_id based on username and tagline
    space = %20
    # = -
    """
    summoner = f"{user_name.replace(" ", "%20")}{tagline.replace("#", "-")}"
    url =  f"{base_url}{summoner_detail_url}{summoner}"
    res = http_client.get(url)
    soup = BeautifulSoup(res.text, "lxml")
    script_raw = soup.find("script", id="__NEXT_DATA__").text
    script_json = json.loads(script_raw)
    return script_json["props"]["pageProps"]["data"]["summoner_id"]

def extract_summoner_details(summer_detail_payload: dict[str, Any]) -> dict[str, int | float | str]:
    """
    send request to api to get player details
    """
    # api_url = f"https://op.gg/api/v1.0/internal/bypass/games/na/summoners/{summoner_id}?&limit={n}&hl=en_US&game_type=soloranked"
    # api_summer_detail_res = http_client.get(api_url)

    games = 0
    wins = 0
    losses = 0
    kills = 0
    deaths = 0
    assists = 0
    positions_played = Counter()
    minion_kill = 0
    neutral_minion_kill = 0
    total_team_kills = 0

    for game in summer_detail_payload["data"]:
        games += 1
        wins += 1 if game["myData"]["stats"]["result"] == "WIN" else 0
        losses += 1 if game["myData"]["stats"]["result"] == "LOSE" else 0
        kills += int(game["myData"]["stats"]["kill"])
        deaths += int(game["myData"]["stats"]["death"])
        assists += int(game["myData"]["stats"]["assist"])
        minion_kill += int(game["myData"]["stats"]["minion_kill"])
        neutral_minion_kill += int(game["myData"]["stats"]["neutral_minion_kill"])
        positions_played.update([game["myData"]["position"]])
        
        # either BLUE or RED
        game_team_key: str = game["myData"]["team_key"]
        team_game_info = game["teams"][0] if game_team_key in game["teams"][0] else game["teams"][1]
        total_team_kills += int(team_game_info["game_stat"]["kill"])


    cs_avg = (minion_kill + neutral_minion_kill) / games
    kills_avg = kills / games
    deaths_avg = deaths / games
    assists_avg = assists / games
    preferred_position = Position[positions_played.most_common(1)[0][0].upper()]
    Kill_participation = (kills + assists) / total_team_kills
    return {
        "games_found": games,
        "wins": wins,
        "losses": losses,
        "kills_avg": kills_avg,
        "deaths_avg": deaths_avg,
        "assists_avg": assists_avg,
        "cs_avg": cs_avg,
        "preferred_position": preferred_position,
        "Kill_participation": Kill_participation
    }

def extract_summoner_games_participants(summer_detail_payload: dict[str, Any]) -> list[Summoner]:
    """
    returns a list of players the summoner in payload has played games with based on games in payload
    """
    players_played_with: list[Summoner] = []
    game_num = 0
    extractor_summoner_id: str = summer_detail_payload["data"][game_num]["myData"]["summoner"]["summoner_id"]
    for game in summer_detail_payload["data"]:
        for player in game["participants"]:
            if player["summoner"]["summoner_id"] != extractor_summoner_id:
                participant = Summoner(
                    username=player["summoner"]["game_name"],
                    tagline=player["summoner"]["tagline"]
                )
                players_played_with.append(participant)
    return players_played_with

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
    summoner.kill_participation = data["Kill_participation"]

def get_summoner_profile_details_past_n_games(summoner_profile_details_url: str, summoner: Summoner, http_client: HttpClient, n: int = 20) -> None:
    """
    url needs to have {} where the variables will be injected to
    mutates summoner object with details and returns list of summoners the player participated the last n solo/duo ranked games
    """
    # api_url = f"https://op.gg/api/v1.0/internal/bypass/games/na/summoners/{summoner_id}?&limit={n}&hl=en_US&game_type=soloranked"
    # api_url = "https://op.gg/api/v1.0/internal/bypass/games/na/summoners/{}?&limit={}&hl=en_US&game_type=soloranked".format(summoner_id, n)
    api_url = summoner_profile_details_url.format(summoner.summoner_id, n)

    # api_summer_detail_res = http_client.get(api_url)
    http_client.get(api_url)

    # if api_summer_detail_res == None:
    #     print(f"Request failed")
    #     return None
    
    # api_summer_detail_json = api_summer_detail_res.json()
    api_summer_detail_json = extract_payload_as_json(http_client)
    summoner_details = extract_summoner_details(api_summer_detail_json)
    update_summoner(summoner, summoner_details)
    players_played_with = extract_summoner_games_participants(api_summer_detail_json)
    players_played_with_str = [f"{player.username}#{player.tagline}" for player in players_played_with]
    summoner.players_played_with = players_played_with_str



def generate_summoner_profile_details_past_n_games(summoner_profile_details_url: str, summoner: Summoner, http_client: HttpClient, n: int = 20):
    api_url = summoner_profile_details_url.format(summoner.summoner_id, n)
    http_client.get(api_url)
