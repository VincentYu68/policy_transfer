from gym.envs.registration import register


register(
    id='DartHopperPT-v1',
    entry_point='policy_transfer.envs.dart:DartHopperEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartHopperSoftPT-v1',
    entry_point='policy_transfer.envs.dart:DartHopperSoftEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartHalfCheetahPT-v1',
    entry_point='policy_transfer.envs.dart:DartHalfCheetahEnv',
    max_episode_steps=1000,
)

register(
    id='DartWalker2dPT-v1',
    entry_point='policy_transfer.envs.dart:DartWalker2dEnv',
    max_episode_steps=1000,
)

register(
    id='HalfCheetahPT-v2',
    entry_point='policy_transfer.envs.mujoco:HalfCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='HopperPT-v2',
    entry_point='policy_transfer.envs.mujoco:HopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='Walker2dPT-v2',
    max_episode_steps=1000,
    entry_point='policy_transfer.envs.mujoco:Walker2dEnv',
)
