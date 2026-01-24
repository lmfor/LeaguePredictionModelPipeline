import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers


# Add Categorical Features
CAT_FEATURES = [
    "league", "patch", "year", "split", "playoffs",
    "side", "position",
    "teamname", "champion", "firstPick",
    "ban1","ban2","ban3","ban4","ban5",
    "pick1","pick2","pick3","pick4","pick5",
]

# Add Early Game Numeric Features
NUM_FEATURES = [
    "goldat10","xpat10","csat10",
    "opp_goldat10","opp_xpat10","opp_csat10",
    "golddiffat10","xpdiffat10","csdiffat10",
    "killsat10","assistsat10","deathsat10",
    "opp_killsat10","opp_assistsat10","opp_deathsat10",
]


LABEL = "result"


class Model:
    def __init__(self, df : pd.DataFrame, seed: int = 42):
        self.seed = seed
        
        
        # only keep if they exist (train based off the features we want)
        self.cat_features = [f for f in CAT_FEATURES if f in df.columns]
        self.num_features = [f for f in NUM_FEATURES if f in df.columns]
        # concat
        self.feature_cols = self.cat_features + self.num_features
        
        # missing?
        missing = (set(CAT_FEATURES + NUM_FEATURES) - set(self.feature_cols))
        if missing: print(f"[INFO] Dropping missing columns: {sorted(missing)}")
        
        # drop rows missing a label or highlighted feature
        self.df = df.dropna(subset=self.feature_cols + [LABEL]).copy()
        self.df[LABEL] = self.df[LABEL].astype('int32') # translate result -> int32
        
        # double enforce dtype
        for f in self.cat_features:
            self.df[f] = self.df[f].astype(str)
        for f in self.num_features:
            self.df[f] = pd.to_numeric(self.df[f], errors='coerce')
            
        self.df = self.df.dropna(subset=self.num_features).copy()
        
        self.df_train = None
        self.df_val = None
        self.df_test = None
        
        self.lookups : dict[str, layers.StringLookup] = {}
        self.normalizer : layers.Normalization | None = None
        self.model : keras.Model | None = None
        
    def split(self, train=0.8, val=0.1):
        """_summary_

        Args:
            train (float, optional): % of data that will go to Training the Model | Defaults to 0.8.
            val (float, optional): % of data that will go to Evaluating the Model |. Defaults to 0.1.
        """
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(self.df)) # indexes
        rng.shuffle(idx) # shuffle indexes
        
        n = len(idx)
        n_train = int(train * n)
        n_val = int(val * n)
        
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]
        
        self.df_train = self.df.iloc[train_idx].reset_index(drop=True)
        self.df_val = self.df.iloc[val_idx].reset_index(drop=True)
        self.df_test = self.df.iloc[test_idx].reset_index(drop=True)
        
        print(
            "Label means (train / val / test)",
            self.df_train[LABEL].mean(),
            self.df_val[LABEL].mean(),
            self.df_test[LABEL].mean(),
        )
        
        return self
        
    
    def build_lookups(self):
        """_summary_
        Build a lookup table for all non-numeric features so model can read them
        """
        if self.df_train is None:
            raise RuntimeError("Call split() before calling build_lookups()")
        
        # build StringLookup for each categorical feature
        for f in self.cat_features:
            vocab = self.df_train[f].astype(str).unique()
            self.lookups[f] = layers.StringLookup(
                vocabulary=vocab,
                mask_token = None,
                num_oov_indices=1
            )
            
        # normalizer for numeric block
        if self.num_features:
            self.normalizer = layers.Normalization()
            self.normalizer.adapt(self.df_train[self.num_features].to_numpy().astype("float32"))
            
        return self
    
    def _df_to_ds(self, dataframe: pd.DataFrame, shuffle: bool, batch_size: int):
        """
        pd.DataFrame --> tf.data.Dataset

        Args:
            dataframe (pd.DataFrame): _description_
            shuffle (bool): _description_
            batch_size (int): _description_
        """
        x = {}
        
        for f in self.cat_features:
            x[f] = dataframe[f].astype(str).values
        for f in self.num_features:
            x[f] = dataframe[f].astype("float32").values
            
        y = dataframe[LABEL].values.astype("float32")
        
        ds = tf.data.Dataset.from_tensor_slices((x,y))
        
        if shuffle:
            ds = ds.shuffle(len(dataframe), seed=self.seed, reshuffle_each_iteration=True)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    def build_model(self, cat_dim:int=8, hidden=(128,64), dropout=0.25, l2=1e-4):
        """
        Args:
            cat_dim (int, optional): _description_. Defaults to 8.
            hidden (tuple, optional): _description_. Defaults to (128,64).
            dropout (float, optional): _description_. Defaults to 0.25.
            l2 (_type_, optional): _description_. Defaults to 1e-4.
        """
        if not self.lookups:
            raise RuntimeError("Call build_lookups() before calling build_model().")
        
        inputs, pieces = [], []
        
        # categorical embeddings
        for f in self.cat_features:
            inp = keras.Input(shape=(1,), name=f, dtype=tf.string)
            ids = self.lookups[f](inp)
            emb = layers.Embedding(self.lookups[f].vocabulary_size(), cat_dim)(ids)
            emb = layers.Reshape((cat_dim,))(emb)
            inputs.append(inp)
            pieces.append(emb)
        
        # numeric block
        if self.num_features:
            num_inp = keras.Input(shape=(len(self.num_features),), name="numeric", dtype=tf.float32)
            inputs.append(num_inp)
            
            xnum = num_inp
            if self.normalizer is not None:
                xnum = self.normalizer(xnum)
            pieces.append(xnum)
            
        x = layers.Concatenate()(pieces) if len(pieces) > 1 else pieces[0]
        
        reg = keras.regularizers.l2(l2)
        
        for h in hidden:
            x = layers.Dense(h, activation='relu', kernel_regularizer=reg)(x)
            x = layers.Dropout(dropout)(x)
            
        out = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=out)
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')],
        )
        
        return self
    
    def fit(self, epochs=60, batch_size=256):
        """
        Args:
            epochs (int, optional): _description_. Defaults to 60.
            batch_size (int, optional): _description_. Defaults to 256.
        """
        
        if self.model is None:
            raise RuntimeError("Call build_model() before fit().")
        
        train_ds = self._df_to_ds(self.df_train, shuffle=True, batch_size=batch_size) # type: ignore
        val_ds = self._df_to_ds(self.df_val, shuffle=False, batch_size=batch_size) # type: ignore
        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=8, restore_best_weights=True),
        ]
        
        return self.model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    
    def evaluate(self, batch_size=256):
        """
        Args:
            batch_size (int, optional): _description_. Defaults to 256.
        """
        if self.model is None:
            raise RuntimeError("Train/build model first!")
        
        test_ds = self._df_to_ds(self.df_test, shuffle=False, batch_size=batch_size) #type: ignore
        return self.model.evaluate(test_ds)
    
    def predict_proba(self, row:dict) -> float:
        """
        Args:
            row (dict): _description_

        Returns:
            float: _description_
        """
        if self.model is None:
            raise RuntimeError("Train/build model first.")
        
        x = {}
        
        for f in self.cat_features:
            if f not in row:
                raise KeyError(f"Missing feature {f} in row dict.")
            x[f] = tf.convert_to_tensor(str(row[f]))
            
        if self.num_features:
            nums = []
            for f in self.num_features:
                if f not in row:
                    raise KeyError(f"Missing feature {f} in row dict.")
                nums.append(float(row[f]))
                
            x['numeric'] = tf.convert_to_tensor([nums], dtype=tf.float32)
        return float(self.model.predict(x, verbose=0)[0][0]) #type: ignore