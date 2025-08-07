import os
import json
import csv
import numpy as np
from typing import Dict, Any, Optional

class TrainingLogger:
    """Enhanced logger for per-iteration metrics"""
    def __init__(self):
        self.logs = {}
    
    def log(self, key_or_dict, value=None):
        if isinstance(key_or_dict, dict):
            for k, v in key_or_dict.items():
                self.logs.setdefault(k, []).append(float(v))
        else:
            self.logs.setdefault(key_or_dict, []).append(float(value))
    
    def get_logs(self):
        return self.logs

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.get_logs(), f, indent=2)
