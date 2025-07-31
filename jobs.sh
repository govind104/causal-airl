#!/bin/bash
# jobs.sh — Sequential job submission with -N and -hold_jid

echo 'Submitting jobs in correct dependency order using job names...'

# Submit each job with a hold on the previous
qsub -N job1_gridworld     jobs/run_gridworld_baselines.qsub
qsub -N job2_airl          -hold_jid job1_gridworld     jobs/run_airl_experiments.qsub
qsub -N job3_confounded    -hold_jid job2_airl          jobs/run_confounded_gridworld.qsub
qsub -N job4_cartpole      -hold_jid job3_confounded    jobs/run_cartpole_comparison.qsub
qsub -N job5_generalise    -hold_jid job4_cartpole      jobs/run_generalisation_test.qsub
qsub -N job6_figures       -hold_jid job5_generalise    jobs/run_generate_figures.qsub

echo "All jobs submitted with explicit dependencies."