#!/bin/bash
# scripts/generate_latest_symlinks.sh
set -euo pipefail

mkdir -p results/latest

# Create symlinks to latest results
find_latest() {
    ls -td results/$1/* 2>/dev/null | head -1
}

link_if_exists() {
    if [ -n "$1" ]; then
        ln -sfn "$1" "results/latest/$(basename $1)"
    fi
}

link_if_exists "$(find_latest gridworld_baselines)"
link_if_exists "$(find_latest confounded)"
link_if_exists "$(find_latest cartpole)"
link_if_exists "$(find_latest airl_ablation)"
link_if_exists "$(find_latest generalization)"

echo "Created symlinks in results/latest/"