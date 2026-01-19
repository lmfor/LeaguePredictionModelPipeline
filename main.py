# Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Files
from src.data import DatasetConfig, RetrieveData

cfg = DatasetConfig(
    dataset="xmorra/lol2020esports",
    file="matches2020.csv",
)

load = RetrieveData(cfg)
df = load.load_df()

print(df.head())