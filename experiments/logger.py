import os
import json
import torch
import numpy as np


class TrainingLogger:
    """Enhanced logger for per-iteration metrics"""
    def __init__(self):
        self.logs = {}

    def log(self, key, value=None):
        """Log a scalar or dictionary of scalar values. Values must be float-compatible."""
        def sanitize(val):
            if isinstance(val, (float, int)):
                return float(val)
            if isinstance(val, torch.Tensor):
                return float(val.item()) if val.numel() == 1 else None
            if isinstance(val, np.ndarray):
                return float(val.item()) if val.size == 1 else None
            return None  # unsupported types (e.g., lists, dicts, None)

        if isinstance(key, dict):
            for k, v in key.items():
                safe_v = sanitize(v)
                if safe_v is not None:
                    self.logs.setdefault(k, []).append(safe_v)
        else:
            safe_v = sanitize(value)
            if safe_v is not None:
                self.logs.setdefault(key, []).append(safe_v)

    def get_logs(self):
        return self.logs

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.get_logs(), f, indent=2)
