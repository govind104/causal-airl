import argparse
import json
import os
import glob
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True, help="result dirs to scan")
    parser.add_argument("--out", type=str, default="results/tables/hparams.csv")
    parser.add_argument("--pattern", type=str, default="**/manifest.json", help="Search pattern for manifest files")
    args = parser.parse_args()

    rows = []
    for root in args.roots:
        search_pattern = os.path.join(root, args.pattern)
        for mj in glob.glob(search_pattern, recursive=True):
            with open(mj) as f:
                man = json.load(f)
            row = {'path': os.path.dirname(mj)}
            row.update({f"override.{k}": v for k,v in (man.get('overrides') or {}).items()})
            row['git_hash'] = man.get('git_hash')
            row['seed'] = man.get('seed')
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out} with {len(df)} rows")


if __name__ == "__main__":
    main()
