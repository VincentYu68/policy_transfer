ENV_NAME=DartWalker2dPT-v1
mpirun  -np 3 python policy_transfer/ppo/run_ppo.py --env $ENV_NAME --batch_size 10000 --max_step 20000000 \
    --train_up True --seed 1 --dyn_params 7 --dyn_params 8 --dyn_params 9 --dyn_params 10 --dyn_params 11 \
    --dyn_params 12 --dyn_params 13 --dyn_params 14 --name 8d --mirror 1
