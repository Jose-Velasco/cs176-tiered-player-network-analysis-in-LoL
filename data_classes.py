from dataclasses import dataclass
from enums import Tier, Position
from typing import Optional

@dataclass
class Summoner:
    username: str
    tagline: str
    tier: Tier
    lp: int
    level: int
    total_wins: int
    total_losses: int
    past_n_games: int = 20
    summoner_id: Optional[str] = None
    past_n_wins: Optional[int] = None
    past_n_losses: Optional[int] = None
    # the below is attributes are based on the last n ranked games
    # kills
    kills: Optional[float] = None
    # deaths
    deaths: Optional[float] = None
    # assists
    assists: Optional[float] = None
    preferred_position: Optional[Position] = None
    # also known as CS
    creep_score: Optional[float] = None

    def get_kda(self) -> str:
        return f"{self.k}/{self.a}/{self.a}"