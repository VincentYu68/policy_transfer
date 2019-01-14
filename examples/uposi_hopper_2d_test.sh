python policy_transfer/uposi/visualize_uposi.py --env DartHopperPT-v1 \
                                                --policy_path data/ppo_DartHopperPT-v10__UP/policy_params.pkl \
                                                --osi_path data/osi_data/_DartHopperPT-v1_\[0,\ 5\]/osi_params.pkl \
                                                --OSI_hist 10 \
                                                --dyn_params 0 \
                                                --dyn_params 5 \
                                                --reset_mp_prob 0.01