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
        # self.league_games = None
        
    def get_csv(self):
        """
        Intializes self.csv so dev can parse
        """
        return self.csv
        
    def get_all_team_names(self):
        """
        Returns a list of all of the teamnames in the dataset
        """
        return self.csv["teamname"].unique()
        
    def get_teams_by_league(self, league_slug : str):
        """
        Returns a comprehensive list of all of the games in a given league (by slug)
        """
        league_games = self.csv[self.csv["league"] == league_slug]
        league_games = league_games[pd.isna(league_games["playername"])].reset_index(drop=True)
        return league_games
    
    def generate_game_by_league(self, league_slug : str):
        """
        Returns a row (game details) based on input league        
        """
        league_games = self.get_teams_by_league(league_slug=league_slug)
        
        # .itertupes is quicker than .iterrows
        for row in league_games.itertuples():
            yield(row)
            
        
        
    
    
    
    
    