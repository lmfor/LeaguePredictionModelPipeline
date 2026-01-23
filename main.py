# Libraries
import pandas as pd
import matplotlib.pyplot as plt
# from keras import models
# import tensorflow as tf

# Files
# from src.data import DatasetConfig, RetrieveData
# from train_model.model import Model
from src.data import LeagueData

lol2025 = LeagueData("data\\2025_LoL_esports_match_data_from_OraclesElixir.csv")

# all_teams = lol2025.get_all_team_names()
# lck_teams = lol2025.get_teams_by_league(league_slug='LCK')

lck_games = lol2025.get_teams_by_league('LCK')

print(lol2025.get_csv().columns.tolist())
# print(lck_games)

        
    