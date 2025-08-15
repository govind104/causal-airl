#!/bin/bash
# experiments.sh — Sequential job submission with hold_jid on .sh scripts directly

echo 'Submitting jobs in dependency order using updated .sh scripts...'

j1=$(qsub -N gridworld_baselines                      scripts/run_gridworld_baselines.sh      | awk '{print $3}')
j2=$(qsub -N airl_experiments           -hold_jid $j1 scripts/run_airl_experiments.sh         | awk '{print $3}')
j3=$(qsub -N causal_airl_experiments    -hold_jid $j2 scripts/run_causal_airl_experiments.sh  | awk '{print $3}')
j4=$(qsub -N confounded_gridworld       -hold_jid $j3 scripts/run_confounded_gridworld.sh     | awk '{print $3}')
j5=$(qsub -N generalisation_test        -hold_jid $j4 scripts/run_generalisation_test.sh      | awk '{print $3}')
j6=$(qsub -N cartpole_comparison        -hold_jid $j5 scripts/run_cartpole_comparison.sh      | awk '{print $3}')

echo "All jobs submitted in dependency order."
