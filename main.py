# Libraries
import pandas as pd
import matplotlib.pyplot as plt
from keras import models
import tensorflow as tf

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
#model = Model(df).split().augment_swap_sides().build_looksups().build_model()
#model.fit(epochs=400)
#print(f"Test: {model.evaluate()}")
#model.save()

# ==== LOAD MODEL AND PREDICT ====
loaded_model = models.load_model("models/draft_winner_tf.keras")

draft = {
    "league": ["LCK"],
    "blueteam": ["KT Rolster"],
    "redteam": ["T1"], 
    "bluetop": ["Rumble"],
    "bluejungle" : ["LeeSin"],
    "bluemid" : ["Ryze"],
    "blueadc" : ["Ashe"],
    "bluesupport" : ["Braum"],
    "redtop": ["Sion"],
    "redjungle" : ["Darius"],
    "redmid" : ["Taliyah"],
    "redadc" : ["Varus"],
    "redsupport" : ["Poppy"],
}

x = {k: tf.constant(v, dtype=tf.string) for k,v in draft.items()}

p = float(loaded_model.predict(x,verbose=0)[0][0]) # type: ignore
print("Blue win proabability: ", p)
print("Predicted winner:", "Blue" if p >= 0.5 else "Red")






