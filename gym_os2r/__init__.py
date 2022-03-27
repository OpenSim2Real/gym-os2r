import numpy
from . import tasks
from . import models
from . import randomizers
from . import common
from . import utils
from . import runtimes

__all__ = ['tasks', 'models', 'randomizers', 'common', 'utils', 'runtimes']

from gym.envs.registration import register
from gym_os2r.rewards import *

max_float = float(numpy.finfo(numpy.float32).max)

register(
    id='Monopod-stand-v1',
    entry_point='gym_os2r.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=100_000,
    kwargs={'task_cls': tasks.monopod.MonopodTask,
            'agent_rate': 1000,
            'physics_rate': 10000,
            'real_time_factor': max_float,
            'task_mode': 'fixed_hip',
            'reward_class': StandingV1,
            'reset_positions': ['ground']
            })
register(
    id='Monopod-balance-v1',
    entry_point='gym_os2r.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=100_000,
    kwargs={'task_cls': tasks.monopod.MonopodTask,
            'agent_rate': 1000,
            'physics_rate': 10000,
            'real_time_factor': max_float,
            'task_mode': 'fixed_hip_simple',
            'reward_class': BalancingV1,
            'reset_positions': ['stand']
            })
register(
    id='Monopod-balance-v2',
    entry_point='gym_os2r.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=100_000,
    kwargs={'task_cls': tasks.monopod.MonopodTask,
            'agent_rate': 1000,
            'physics_rate': 10000,
            'real_time_factor': max_float,
            'task_mode': 'fixed_hip_simple',
            'reward_class': BalancingV2,
            'reset_positions': ['stand']
            })

register(
    id='Monopod-balance-v3',
    entry_point='gym_os2r.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=10_000,
    kwargs={'task_cls': tasks.monopod.MonopodTask,
            'agent_rate': 1000,
            'physics_rate': 10000,
            'real_time_factor': max_float,
            'task_mode': 'fixed_hip_simple',
            'reward_class': BalancingV2,
            'reset_positions': ['stand', 'half_stand', 'ground', 'lay', 'float']
            })

register(
    id='Monopod-nonorm-balance-v1',
    entry_point='gym_os2r.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=100_000,
    kwargs={'task_cls': tasks.monopod_no_norm.MonopodTask,
            'agent_rate': 1000,
            'physics_rate': 10000,
            'real_time_factor': max_float,
            'task_mode': 'fixed_hip_simple',
            'reward_class': BalancingV1,
            'reset_positions': ['stand']
            })
register(
    id='Monopod-nonorm-balance-v2',
    entry_point='gym_os2r.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=100_000,
    kwargs={'task_cls': tasks.monopod_no_norm.MonopodTask,
            'agent_rate': 1000,
            'physics_rate': 10000,
            'real_time_factor': max_float,
            'task_mode': 'fixed_hip_simple',
            'reward_class': BalancingV2,
            'reset_positions': ['stand']
            })

register(
    id='Monopod-nonorm-balance-v3',
    entry_point='gym_os2r.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=10_000,
    kwargs={'task_cls': tasks.monopod_no_norm.MonopodTask,
            'agent_rate': 1000,
            'physics_rate': 10000,
            'real_time_factor': max_float,
            'task_mode': 'fixed_hip_simple',
            'reward_class': BalancingV2,
            'reset_positions': ['stand', 'half_stand', 'ground', 'lay', 'float']
            })

register(
    id='Monopod-hop-v1',
    entry_point='gym_os2r.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=100_000,
    kwargs={'task_cls': tasks.monopod.MonopodTask,
            'agent_rate': 1000,
            'physics_rate': 10000,
            'real_time_factor': max_float,
            'task_mode': 'free_hip',
            'reward_class': HoppingV1,
            'reset_positions': ['stand']
            })

register(
    id='Monopod-simple-v1',
    entry_point='gym_os2r.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=100_000,
    kwargs={'task_cls': tasks.monopod.MonopodTask,
            'agent_rate': 1000,
            'physics_rate': 10_000,
            'real_time_factor': max_float,
            'task_mode': 'simple',
            'reward_class': StraightV1,
            'reset_positions': ['stand']
            })
