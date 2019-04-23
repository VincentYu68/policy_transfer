from policy_transfer.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from policy_transfer.envs.dart.parameter_managers import *

from policy_transfer.envs.mujoco.half_cheetah import HalfCheetahEnv
from policy_transfer.envs.mujoco.hopper import HopperEnv
from policy_transfer.envs.mujoco.walker2d import Walker2dEnv