ENV_NAME=DartHopperPT-v1
mpirun  -np 3 python policy_transfer/ppo/run_ppo.py --env $ENV_NAME --batch_size 4000 --max_step 10000000 \
    --train_up True --seed 0 --dyn_params 0 --dyn_params 1 --dyn_params 2 --dyn_params 5 --dyn_params 9 --name 5d_2
