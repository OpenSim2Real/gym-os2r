import numpy
from . import tasks
from . import models
from . import randomizers
from . import monitor
from . import common
from . import utils

__all__ = ['tasks', 'models', 'randomizers', 'monitor', 'common', 'utils']

from gym.envs.registration import register


max_float = float(numpy.finfo(numpy.float32).max)

register(
    id='Monopod-v1',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=5000,
    kwargs={'task_cls': tasks.monopod.MonopodTask,
            'agent_rate': 1000,
            'physics_rate': 1000,
            'real_time_factor': max_float,
            'supported_models': ['monopod_v1'],
            'task_mode': 'free_hip'
            })
register(
    id='Monopod-fh-v1',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=5000,
    kwargs={'task_cls': tasks.monopod.MonopodTask,
            'agent_rate': 1000,
            'physics_rate': 1000,
            'real_time_factor': max_float,
            'supported_models': ['monopod_v1_fh'],
            'task_mode': 'fixed_hip'
            })

register(
    id='Monopod-fh-fby-v1',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=5000,
    kwargs={'task_cls': tasks.monopod.MonopodTask,
            'agent_rate': 1000,
            'physics_rate': 1000,
            'real_time_factor': max_float,
            'supported_models': ['monopod_v1_fh_fby'],
            'task_mode': 'fixed_hip_and_boom_yaw'
            })
