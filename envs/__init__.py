from gym.envs.registration import register


register(
    id='DartHopper-v1',
    entry_point='envs.dart:DartHopperEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartHopperSoft-v1',
    entry_point='envs.dart:DartHopperSoftEnv',
    reward_threshold=3800.0,
    max_episode_steps=1000,
)

register(
    id='DartHalfCheetah-v1',
    entry_point='envs.dart:DartHalfCheetahEnv',
    max_episode_steps=1000,
)

register(
    id='DartWalker2d-v1',
    entry_point='envs.dart:DartWalker2dEnv',
    max_episode_steps=1000,
)

register(
    id='HalfCheetah-v2',
    entry_point='envs.mujoco:HalfCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='Hopper-v2',
    entry_point='envs.mujoco:HopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='Walker2d-v2',
    max_episode_steps=1000,
    entry_point='envs.mujoco:Walker2dEnv',
)
