ENV_NAME=DartHopperPT-v1
mpirun  -np 4 python policy_transfer/meta_strategy_optimization/run_mso_ars.py --env $ENV_NAME --task_number 8 \
    --skilldim 2 --seed 0 --dyn_params 0 --dyn_params 1 --dyn_params 2 --dyn_params 5 --dyn_params 9 --name 5d
