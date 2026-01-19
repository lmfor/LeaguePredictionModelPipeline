# Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Files
from src.data import DatasetConfig, RetrieveData

matches_cfg = DatasetConfig(
    dataset="xmorra/lol2020esports",
    file="matches2020.csv",
)

champs_cfg = DatasetConfig(
    dataset="xmorra/lol2020esports",
    file="champion_stats.csv"
)

matches_loader = RetrieveData(matches_cfg)
champs_loader = RetrieveData(champs_cfg)

matches_df = matches_loader.load_df()
champs_df = champs_loader.load_df()

"""
====== NOTES =====

For Matches:
0 -> Blue Side win
1 -> Red Side win

There are NO duplicate gameids!
"""

