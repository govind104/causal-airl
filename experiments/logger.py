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
            if val is None:
                return None
            if isinstance(val, torch.Tensor):
                if val.numel() == 1:
                    return float(val.item())
                else:
                    # Convert multi-dimensional tensors to lists instead of dropping
                    return val.detach().cpu().numpy().tolist()
            if isinstance(val, np.ndarray):
                if val.size == 1:
                    return float(val.item())
                else:
                    # Convert multi-dimensional arrays to lists instead of dropping
                    return val.tolist()
            if isinstance(val, str):
                return val  # Add string support
            return None  # unsupported types

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
