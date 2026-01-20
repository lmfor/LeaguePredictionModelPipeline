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
# cfg = DatasetConfig(dataset="xmorra/lol2020esports",file="matches2020.csv")
# df = RetrieveData(cfg).load_df()

def get_user_input():
    print("\n--- LoL WinSide Predictor ---")
    print("Please enter the draft details")
    
    features = [
    "league",
    "blueteam", "redteam",
    "bluetop", "bluejungle", "bluemid", "blueadc", "bluesupport",
    "redtop", "redjungle", "redmid", "redadc", "redsupport"]
    
    user_draft = {}
    
    for feature in features:
        val = input(f"{feature.capitalize()}: ").strip()
        user_draft[feature] = val
        
    return user_draft

# Load Model
MODEL_PATH = 'models/draft_winner_tf.keras'
import os
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR]: Model not found at {MODEL_PATH}")
    exit(1)
    
loaded_model = models.load_model(MODEL_PATH)

while True:
    draft = get_user_input()
    
    x = {k: tf.constant([v], dtype=tf.string) for k, v in draft.items()}  # batch size 1
    y = loaded_model.predict(x, verbose=0) # type: ignore
    prediction = float(tf.reshape(y, [-1])[0])
    
    print("\n" + "="*30)
    print(f"Blue Win Probability: {prediction:.2%}")
    print(f"Predicted Winner: {'BLUE' if prediction >= 0.5 else 'RED'}")
    print("="*30)
    
    cont = input("\nPredict another match? (y/n): ").lower()
    if cont != 'y':
        break


# =================== LEGACY =================== #

# ==== INITIALIZE AND FIT MODEL ====
#model = Model(df).split().augment_swap_sides().build_looksups().build_model()
#model.fit(epochs=400)
#print(f"Test: {model.evaluate()}")
#model.save()

# ==== LOAD MODEL AND PREDICT ====
#loaded_model = models.load_model("models/draft_winner_tf.keras")

## 2025 Worlds Finals T1 vs KT Game 1
#draft_1  = {
#    "league": ["Riot"],
#    "blueteam": ["KT Rolster"],
#    "redteam": ["T1"], 
#    "bluetop": ["Rumble"],
#    "bluejungle" : ["LeeSin"],
#    "bluemid" : ["Ryze"],
#    "blueadc" : ["Ashe"],
#    "bluesupport" : ["Braum"],
#    "redtop": ["Sion"],
#    "redjungle" : ["Darius"],
#    "redmid" : ["Taliyah"],
#    "redadc" : ["Varus"],
#    "redsupport" : ["Poppy"],
#}

## 2021 LCK Summer Finals T1 vs DK Game 2
#draft_2 = {
#    "league": ["LCK"],
#    "blueteam": ["DAMWON Gaming"],
#    "redteam": ["T1"], 
#    "bluetop": ["Camille"],
#    "bluejungle" : ["Olaf"],
#    "bluemid" : ["Kassadin"],
#    "blueadc" : ["Ziggs"],
#    "bluesupport" : ["Leona"],
#    "redtop": ["Ornn"],
#    "redjungle" : ["Khazix"],
#    "redmid" : ["Leblanc"],
#    "redadc" : ["Vayne"],
#    "redsupport" : ["Trundle"],
#}

#x1 = {k: tf.constant(v, dtype=tf.string) for k,v in draft_1.items()}
#x2 = {k: tf.constant(v, dtype=tf.string) for k,v in draft_2.items()}

#p = float(loaded_model.predict(x,verbose=0)[0][0]) # type: ignore
#print("Blue win proabability: ", p)
#print("Predicted winner:", "Blue" if p >= 0.5 else "Red")






