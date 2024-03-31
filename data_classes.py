from dataclasses import dataclass
from enums import Tier, Position
from typing import Optional

@dataclass
class Summoner:
    username: str
    tagline: str
    tier: Optional[Tier] = None
    lp: Optional[int] = None
    level: Optional[int] = None
    total_wins: Optional[int] = None
    total_losses: Optional[int] = None
    games_found: Optional[int] = None
    summoner_id: Optional[str] = None
    past_n_wins: Optional[int] = None
    past_n_losses: Optional[int] = None
    damage_dealt_to_champions: Optional[float] = None
    damage_taken: Optional[float] = None
    control_wards_placed: Optional[float] = None
    wards_placed: Optional[float] = None
    wards_kills: Optional[float] = None
    # kills
    kills: Optional[float] = None
    # deaths
    deaths: Optional[float] = None
    # assists
    assists: Optional[float] = None
    preferred_position: Optional[Position] = None
    kill_participation: Optional[float] = None
    # also known as CS
    creep_score: Optional[float] = None
    players_played_with: Optional[list[str]] = None

    def get_kda(self) -> str:
        return f"{self.kills}/{self.deaths}/{self.assists}"