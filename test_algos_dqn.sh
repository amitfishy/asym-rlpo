#!/bin/bash

export WANDB_MODE=dryrun
export WANDB_CONSOLE=off
export WANDB_SILENT=true

envs=(PO-pos-CartPole-v1 ../gym-gridverse/yaml/gv_empty.4x4.yaml)
algos=(fob-dqn foe-dqn poe-dqn poe-adqn)

args=(
  --episode-buffer-prepopulate-timesteps 100 --max-simulation-timesteps 500
)

for env in ${envs[@]}; do
  for algo in ${algos[@]}; do
    echo ./main_dqn.py $env $algo ${args[@]}
    python -W ignore ./main_dqn.py $env $algo ${args[@]} > /dev/null
    [ $? -eq 0 ] && echo "SUCCESS" || echo "FAIL"
  done
done

exit 0