import os
import json
import csv
import numpy as np
from typing import Dict, Any, Optional

class TrainingLogger:
    """Enhanced logger for per-iteration metrics"""
    def __init__(self):
        self.logs = {}
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if step is None:
            step = len(self.logs)
        self.logs[step] = metrics
    
    def get_logs(self):
        return {
            step: {k: float(v) if isinstance(v, (np.floating, float)) else v
                   for k, v in metrics.items()}
            for step, metrics in self.logs.items()
        }

def save_json_log(data: Dict[str, Any], save_path: str):
    """Save evaluation results or run metadata as JSON."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)

def save_csv_row(data: Dict[str, Any], csv_path: str):
    """
    Append a row to a summary CSV. Creates header if new file.
    """
    # Convert None to empty string for CSV
    cleaned_data = {k: v if v is not None else '' for k, v in data.items()}
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=cleaned_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(cleaned_data)

def summarise_run(
    metrics: Dict[str, float],
    config: Dict[str, Any],
    save_dir: str
):
    """
    Log all run results into:
    - JSON (full metrics)
    - CSV (summary row)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save full config + metrics
    log_path = os.path.join(save_dir, "results.json")
    save_json_log({
        "config": config,
        "metrics": metrics
    }, log_path)

    # Flatten + write summary CSV row
    flat_row = {
        "method": config["irl"]["method"],
        "gamma": config["irl"]["gamma"],
        "demos": config["expert"]["num_trajectories"],
        "slip": config["env"]["slip_prob"],
        "reward_corr": round(metrics.get("reward_correlation", -1), 4),
        "value_diff": round(metrics.get("value_difference", -1), 4),
        "policy_agreement": round(metrics.get("policy_agreement", -1), 4),
    }
    save_csv_row(flat_row, os.path.join(save_dir, "summary.csv"))
