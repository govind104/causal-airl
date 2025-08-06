import os
import json
import csv
import numpy as np
from typing import Dict, Any, Optional

class TrainingLogger:
    """Enhanced logger for per-iteration metrics"""
    def __init__(self):
        self.logs = {}
    
    def log(self, key: str, value: float):
        self.logs.setdefault(key, []).append(float(value))
    
    def get_logs(self):
        return self.logs

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.get_logs(), f, indent=2)
