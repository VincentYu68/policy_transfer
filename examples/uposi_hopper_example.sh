mpirun -np 4 python ppo.run_ppo --env DartHopper-v1 --batch_size 2500 --max_step 5000000 --train_up True --seed 0

python uposi.train_osi --env DartHopper-v1 --policy_path data/ppo_DartHopper-v10__UP/poicy_params.pkl --dyn_params 0 --dyn_params 1

