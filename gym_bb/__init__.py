import numpy
from . import tasks
from . import models
from . import randomizers
from . import common
from . import utils
from . import runtime

__all__ = ['tasks', 'models', 'randomizers', 'common', 'utils', 'runtime']

from gym.envs.registration import register
from gym_bb.rewards import BalancingV1, StandingV1, WalkingV1

max_float = float(numpy.finfo(numpy.float32).max)

register(
    id='Monopod-stand-v1',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=1000,
    kwargs={'task_cls': tasks.monopod.MonopodTask,
            'agent_rate': 1000,
            'physics_rate': 5000,
            'real_time_factor': max_float,
            'task_mode': 'free_hip',
            'reward_class': StandingV1,
            'reset_positions': ['ground']
            })
register(
    id='Monopod-balance-v1',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=1000,
    kwargs={'task_cls': tasks.monopod.MonopodTask,
            'agent_rate': 1000,
            'physics_rate': 5000,
            'real_time_factor': max_float,
            'task_mode': 'free_hip',
            'reward_class': BalancingV1,
            'reset_positions': ['stand']
            })

register(
    id='Monopod-walk-v1',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=1000,
    kwargs={'task_cls': tasks.monopod.MonopodTask,
            'agent_rate': 1000,
            'physics_rate': 5000,
            'real_time_factor': max_float,
            'task_mode': 'free_hip',
            'reward_class': WalkingV1,
            'reset_positions': ['stand']
            })
#
# register(
#     id='Monopod-v1',
#     entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
#     max_episode_steps=5000,
#     kwargs={'task_cls': tasks.monopod.MonopodTask,
#             'agent_rate': 1000,
#             'physics_rate': 1000,
#             'real_time_factor': max_float,
#             })
# register(
#     id='Monopod-fixed_hip-v1',
#     entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
#     max_episode_steps=5000,
#     kwargs={'task_cls': tasks.monopod.MonopodTask,
#             'agent_rate': 1000,
#             'physics_rate': 1000,
#             'real_time_factor': max_float,
#             'task_mode': 'fixed_hip'
#             })
#
# register(
#     id='Monopod-fixed_hip_and_boom_yaw-v1',
#     entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
#     max_episode_steps=5000,
#     kwargs={'task_cls': tasks.monopod.MonopodTask,
#             'agent_rate': 1000,
#             'physics_rate': 1000,
#             'real_time_factor': max_float,
#             'task_mode': 'fixed_hip_and_boom_yaw'
#             })
