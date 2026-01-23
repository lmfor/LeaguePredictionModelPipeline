import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
from keras.optimizers import Adam


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
        self.df = df.dropna(subset=FEATURE_COLS + [LABEL]).copy() # delete row if it has ANY empty values
        self.df[LABEL] = self.df[LABEL].astype("int32")
        
        # INIT as NONE for error control
        self.team_lookup = None
        self.champ_lookup = None
        self.model = None
        
        self.df_train = None
        self.df_val = None
        self.df_test = None
        
    # augment side swaps
    def augment_swap_sides(self):
        if self.df_train is None:
            raise RuntimeError("Call split() before augment_swap_sides().")
        
        df = self.df_train.copy()
        
        # swap teams
        swap = df.copy()
        swap["blueteam"], swap["redteam"] = df["redteam"], df["blueteam"]
        
        # swamp champs
        blue_cols = ["bluetop", "bluejungle", "bluemid", "blueadc", "bluesupport"]
        red_cols = ["redtop", "redjungle", "redmid", "redadc", "redsupport"]
        
        for b, r in zip(blue_cols, red_cols):
            swap[b] = df[r]
            swap[r] = df[b]
            
        # flip the label(s)
        swap["result"] = 1 - df["result"]
        
        self.df_train = pd.concat([swap, df], ignore_index=True)
        return self
        
    
    # Split into training & eval datasets
    def split(self, train=0.8, val=0.1):
        
        # ==========
        # Shuffle the data
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(self.df))
        rng.shuffle(idx)
        
        n = len(idx)
        n_train = int(train * n)
        n_val = int(val * n)
        # ==========
        
        # Allocating training / val / test data
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train+n_val]
        test_idx = idx[n_train + n_val:]
        
        # initialize vals
        self.df_train = self.df.iloc[train_idx].reset_index(drop=True)
        self.df_val = self.df.iloc[val_idx].reset_index(drop=True)
        self.df_test = self.df.iloc[test_idx].reset_index(drop=True)
        
        print(
            "Label means (train / val / test):",
            self.df_train['result'].mean(),
            self.df_val["result"].mean(),
            self.df_test["result"].mean(),
        )
        
        return self

        
        
    
    # build vocab + lookups
    def build_looksups(self):
        if self.df_train is None:
            raise RuntimeError("Call split() before build_lookups()")

        # ==
        teams = pd.concat([self.df_train["blueteam"], self.df_train["redteam"]]).astype(str).unique()
       
        champs = pd.concat([
            self.df_train["bluetop"], self.df_train["bluejungle"], self.df_train["bluemid"],
            self.df_train["blueadc"], self.df_train["bluesupport"],
            self.df_train["redtop"], self.df_train["redjungle"], self.df_train["redmid"],
            self.df_train["redadc"], self.df_train["redsupport"],
        ], axis=0).astype(str).unique()
        
        self.team_lookup = layers.StringLookup(vocabulary=teams, mask_token=None, num_oov_indices=1)
        self.champ_lookup = layers.StringLookup(vocabulary=champs, mask_token=None, num_oov_indices=1)
        return self
    
    def _df_to_ds(
            self, 
            dataframe : pd.DataFrame, 
            shuffle : bool, 
            batch_size: int):
        
        x = {col: dataframe[col].astype(str).values for col in FEATURE_COLS}
        y = dataframe[LABEL].values
        ds = tf.data.Dataset.from_tensor_slices((x,y))
        
        if shuffle:
            ds = ds.shuffle(len(dataframe), seed=self.seed, reshuffle_each_iteration=True)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    
    # Construct model
    def build_model(
            self, 
            team_dim : int = 8, 
            champ_dim: int = 16, 
            hidden=(128,68), 
            dropout=0.25):
        
        if self.team_lookup is None or self.champ_lookup is None:
            raise RuntimeError("Call build_lookups() before calling build_model()")
        
        def embed_input(name: str, lookup: layers.StringLookup, dim: int):
            inp = keras.Input(shape=(1,), name=name, dtype=tf.string)
            ids = lookup(inp)
            emb = layers.Embedding(input_dim=lookup.vocabulary_size(), output_dim=dim)(ids)
            emb = layers.Reshape((dim,))(emb)
            
            return inp, emb
        
        inputs, embs = [], []
        
        league_vocab = self.df_train["league"].astype(str).unique() #type: ignore
        self.league_lookup = layers.StringLookup(vocabulary=league_vocab, mask_token=None, num_oov_indices=1)
        inp, emb = embed_input("league", self.league_lookup, dim=4)
        inputs.append(inp)
        embs.append(emb)
        
        for col in ["blueteam", "redteam"]:
            inp, emb = embed_input(col, self.team_lookup, team_dim)
            inputs.append(inp)
            embs.append(emb)
            
        for col in [
            "bluetop", "bluejungle", "bluemid", "blueadc", "bluesupport",
            "redtop", "redjungle", "redmid", "redadc", "redsupport", ]:
            
            inp, emb = embed_input(col, self.champ_lookup, champ_dim)
            inputs.append(inp)
            embs.append(emb)
        
        reg = keras.regularizers.l2(1e-4)

        x = layers.Concatenate()(embs)

        x = layers.Dense(64, activation="relu", kernel_regularizer=reg)(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(32, activation="relu", kernel_regularizer=reg)(x)
        x = layers.Dropout(0.5)(x)

        out = layers.Dense(1, activation="sigmoid")(x)
        
        self.model = keras.Model(inputs=inputs, outputs=out)
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        return self
            
    
    # train & eval & save
    def fit(self, epochs=60, batch_size=128):
        if self.model is None:
            raise RuntimeError("Call build_model() before fit()")
        
        train_ds = self._df_to_ds(self.df_train, shuffle=True, batch_size=batch_size) # type: ignore
        val_ds = self._df_to_ds(self.df_val, shuffle=False, batch_size=batch_size) # type: ignore
        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True),
        ]
        
        return self.model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    
    
    def evaluate(self, batch_size=128):
        
        test_ds = self._df_to_ds(self.df_test, shuffle=False, batch_size=batch_size) # type: ignore
        return self.model.evaluate(test_ds) # type: ignore
    
    def save(self, path : str = "models/draft_winner_tf.keras"):
        if self.model is None:
            raise RuntimeError("[SAVE ERROR] ! No model to save.")
        
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        return path
    
    
    # Prediction
    def predict_blue_side_win(self, row:dict) -> float:
        if self.model is None:
            raise RuntimeError("[PREDICTION ERROR] ! Train/build model first.")
        
        x = {k: tf.convert_to_tensor([str(row[k])]) for k in FEATURE_COLS}
        probability_blue_win = float(self.model.predict(x, verbose=0)[0][0]) #type: ignore
        
        return probability_blue_win
        
    
        
