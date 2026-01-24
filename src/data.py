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
        Return type: pd.DataFrame  
        """
        league_games = self.get_teams_by_league(league_slug=league_slug)
        
        # .itertupes is quicker than .iterrows
        for row in league_games.itertuples(index=False):
            yield(row)
            
    def print_row_features(self, row, title: str = "GAME ROW"):
        """ Comprehensive
        Args:
            row (pd.DataFrame): Row Focused
            title (str, optional): Row Title | Defaults to "GAME ROW".
        """
        d = row._asdict()
        max_key_len = max(len(k) for k in d.keys())
        line = "=" * (max_key_len + 40)
        
        print(line)
        print(f"{title}".center(len(line)))
        print(line)
        
        for k, v in d.items():
            print(f"{k:<{max_key_len}} : {v}")
            
        print(line)
    
    def print_row_summary(self, row, title: str = "GAME ROW"):
        """ Summary
        Args:
            row (pd.DataFrame): Row Focused
            title (str, optional): Row Title | Defaults to "GAME ROW".
        """
        d = row._asdict()
        keys = ["gameid", "league", "teamname", "side", "playername", "result", "gamelength", "kills", "deaths", "assists"]
        for key in keys:
            print(f"{key}: {d.get(key)}")
            
        
        
    
    
    
    
    