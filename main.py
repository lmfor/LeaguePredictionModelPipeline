# Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Files
from src.data import DatasetConfig, RetrieveData
from tflow.model import Model

"""
champs_cfg = DatasetConfig(
    dataset="xmorra/lol2020esports",
    file="champion_stats.csv"
)
"""

# ==== IMPORT DATA & INIT DATAFRAME ====
cfg = DatasetConfig(dataset="xmorra/lol2020esports",file="matches2020.csv")
df = RetrieveData(cfg).load_df()

# ==== INITIALIZE AND FIT MODEL ====
# model = Model(df).split()build_looksups().build_model()






