from enum import StrEnum

class Tier(StrEnum):
    CHALLENGER = "challenger"
    GRANDMASTER = "grandmaster"
    MASTER = "master"

class Position(StrEnum):
    TOP = "top"
    JUNGLE = "jungle"
    MID = "mid"
    ADC = "adc"
    SUPPORT = "support"