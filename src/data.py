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
    
    
class LoadCSV():
    def __init__(self, local_path : str):
        self.csv = pd.read_csv(local_path)
        
    def _get_team_names(self):
        return self.csv["teamname"].unique()
        
    