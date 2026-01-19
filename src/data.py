from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict, List
import pandas as pd
from pathlib import Path

@dataclass(frozen=True)
class DatasetConfig:
    """
    Dataset Parameters
    """
    
    dataset : str                       # url
    file: Optional[str] = None          # "".csv
    cache_dir : str = "data"            # Storage dir
    
    

class RetrieveData:
    """
    Downloads and loads a KaggleHub dataset.
    Returns a pandas DF or Path
    """
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.cache_dir = Path(cfg.cache_dir)
        
    def download(self) -> Path:
        import kagglehub
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)
            
        # kagglehub ds returns a local string path
        ds_path = Path(kagglehub.dataset_download(self.cfg.dataset))
            
        return ds_path
    
    def load_df(self) -> pd.DataFrame:
        ds_path = self.download()
        
        if not self.cfg.file:
            raise ValueError("You must set cfg.file='____.csv")
        
        file_path = ds_path / self.cfg.file
        return pd.read_csv(file_path)
        