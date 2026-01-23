# Libraries
import pandas as pd
import matplotlib.pyplot as plt
# from keras import models
# import tensorflow as tf

# Files
# from src.data import DatasetConfig, RetrieveData
# from train_model.model import Model
from src.data import LoadCSV

lol2026 = LoadCSV("data\\2026_LoL_esports_match_data_from_OraclesElixir.csv")


print(lol2026._get_team_names())
