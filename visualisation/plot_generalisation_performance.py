import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def plot_generalization_comparison(df, metrics, save_path=None):
    plt.figure(figsize=(10, 6))
    palette = {"airl": "orange", "causal_airl": "blue"}
    
    for metric in metrics:
        plt.subplot(1, len(metrics), metrics.index(metric)+1)
        ax = sns.barplot(
            x="train_z", 
            y=metric, 
            hue="method", 
            data=df, 
            palette=palette,
            errorbar="sd"
        )
        plt.title(metric.replace("_", " ").title())
        plt.xlabel("Training Z")
        plt.ylabel("")
        plt.ylim(0, 1)
        plt.legend().remove()
    
    plt.suptitle("Generalization to Z=1 Performance", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved generalization plot to {save_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot generalization performance")
    parser.add_argument("--csv", type=str, required=True, help="CSV with generalization results")
    parser.add_argument("--save_path", type=str, required=True, help="Output path for plot")
    parser.add_argument("--metrics", type=str, default="reward_corr,policy_agreement", help="Comma-separated metrics")
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv)
    metrics = args.metrics.split(",")
    
    # Filter to only test results on Z=1
    plot_df = df[df["test_z"] == 1].copy()
    
    plot_generalization_comparison(plot_df, metrics, args.save_path)