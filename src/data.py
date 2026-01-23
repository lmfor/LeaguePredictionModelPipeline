from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict, List
import pandas as pd
from pathlib import Path

"""
@dataclass(frozen=True)
class DatasetConfig:
    dataset : str                       # url
    file: Optional[str] = None          # "".csv
    cache_dir : str = "data"            # Storage dir
"""
    
class LeagueData():
    def __init__(self, local_path : str):
        self.csv = pd.read_csv(local_path)
        # self.team = None
        
    def get_csv(self):
        return self.csv
        
    def get_all_team_names(self):
        return self.csv["teamname"].unique()
        
    def get_teams_by_league(self, league_slug : str):
        league_games = self.csv[self.csv["league"] == league_slug]
        league_games = league_games[pd.isna(league_games["playername"])].reset_index(drop=True)
        return league_games
    
    
    
    
    