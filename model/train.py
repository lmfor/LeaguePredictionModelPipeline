import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
from src.data import DatasetConfig, RetrieveData


# Prepare df

FEATURE_COLS = [
    "league",
    "blueteam", "redteam",
    "bluetop", "bluejungle", "bluemid", "blueadc", "bluesupport",
    "redtop", "redjungle", "redmid", "redadc", "redsupport",
]

LABEL = "result"


class Model:
    def __init__(self, df : pd.DataFrame, seed: int = 42):
        self.seed = seed
        
        # keep what i need
        self.df = df.dropna(subset=FEATURE_COLS + [LABEL]).copy()
        self.df[LABEL] = self.df[LABEL].astype("int32")
        
        # setup later
        self.team_lookup = None
        self.champ_lookup = None
        self.model = None
        
        self.df_train = None
        self.df_val = None
        self.df_test = None
    
