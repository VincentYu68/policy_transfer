if false; then
python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env HopperPT-v2 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v14_5d_UP_perttorso_scratch_UP/policy_params.pkl --run_cma True
python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env HopperPT-v2 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v13_5d_UP_new_perttorso_UP/policy_params.pkl --run_cma True
python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env HopperPT-v2 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v12_5d_UP_perttorso_scratch_UP/policy_params.pkl --run_cma True

python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env HopperPT-v2 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v14_5d_UP_new_UP/policy_params.pkl --run_cma True
python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env HopperPT-v2 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v13_5d_UP_new_UP/policy_params.pkl --run_cma True
python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env HopperPT-v2 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v12_5d_UP_new2_UP/policy_params.pkl --run_cma True

python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env HopperPT-v2 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v14_5d_UP_new2_pertthigh_scratch_UP/policy_params.pkl --run_cma True
python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env HopperPT-v2 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v13_5d_UP_new2_pertthigh_scratch_UP/policy_params.pkl --run_cma True
python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env HopperPT-v2 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v12_5d_UP_perttihgh_scratch_UP/policy_params.pkl --run_cma True

python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env HopperPT-v2 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v14_5d_UP_pertshin_scratch_UP/policy_params.pkl --run_cma True
python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env HopperPT-v2 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v13_5d_UP_pertshin_scratch_UP/policy_params.pkl --run_cma True
python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env HopperPT-v2 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v12_5d_UP_pertshin_scratch_UP/policy_params.pkl --run_cma True

python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env HopperPT-v2 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v14_5d_UP_new2_pertfoot_scratch_UP/policy_params.pkl --run_cma True
python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env HopperPT-v2 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v13_5d_UP_new2_pertfoot_scratch_UP/policy_params.pkl --run_cma True
python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env HopperPT-v2 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v12_5d_UP_new2_pertfoot_scratch_UP/policy_params.pkl --run_cma True

python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env HopperPT-v2 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v12_5d_UP_new_wideinitialization_UP/policy_params.pkl --run_cma True
python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env HopperPT-v2 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v13_5d_UP_new_wideinitialization_UP/policy_params.pkl --run_cma True
python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env HopperPT-v2 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v14_5d_UP_new_wideinitialization_UP/policy_params.pkl --run_cma True


python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env HopperPT-v2 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v12_5d_UP_new2_UP/policy_params.pkl --run_cma True --robust_policy data/ppo_DartHopperPT-v12_5d_UP_new_pertshin_UP_refpol/policy_params.pkl
python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env HopperPT-v2 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v13_5d_UP_new_UP/policy_params.pkl --run_cma True --robust_policy data/ppo_DartHopperPT-v13_5d_UP_new_pertshin_UP_refpol/policy_params.pkl
python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env HopperPT-v2 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v14_5d_UP_new_UP/policy_params.pkl --run_cma True --robust_policy data/ppo_DartHopperPT-v14_5d_UP_new_pertshin_UP_refpol/policy_params.pkl
fi


python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env DartHopperPT-v1 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v12_5d_UP_new2_UP/policy_params.pkl --run_cma True --robust_policy data/ppo_DartHopperPT-v12_5d_UP_new_2delay_UP_refpol/policy_params.pkl
python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env DartHopperPT-v1 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v13_5d_UP_new_UP/policy_params.pkl --run_cma True --robust_policy data/ppo_DartHopperPT-v13_5d_UP_new_2delay_UP_refpol/policy_params.pkl
python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env DartHopperPT-v1 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v14_5d_UP_new_UP/policy_params.pkl --run_cma True --robust_policy data/ppo_DartHopperPT-v14_5d_UP_new_2delay_UP_refpol/policy_params.pkl

python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env DartHopperPT-v1 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v14_5d_UP_new_UP/policy_params.pkl --run_cma True
python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env DartHopperPT-v1 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v13_5d_UP_new_UP/policy_params.pkl --run_cma True
python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env DartHopperPT-v1 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v12_5d_UP_new2_UP/policy_params.pkl --run_cma True

python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env DartHopperPT-v1 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v12_5d_UP_new_2delay_UP/policy_params.pkl --run_cma True
python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env DartHopperPT-v1 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v13_5d_UP_new_2delay_UP/policy_params.pkl --run_cma True
python policy_transfer/policy_transfer_strategy_optimization/test_socma.py --env DartHopperPT-v1 --name dart2mujoco_5d_thighpert2 --UP_dim 5 --policy_path data/ppo_DartHopperPT-v14_5d_UP_new_2delay_UP/policy_params.pkl --run_cma True