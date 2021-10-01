import numpy
from . import tasks
from . import models
from . import randomizers
from . import monitor
from . import common
from gym.envs.registration import register


max_float = float(numpy.finfo(numpy.float32).max)

register(
    id='Monopod-v1.0.0',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=5000,
    kwargs={'task_cls': tasks.monopod_v1_0_0_balancing.MonopodBalancing,
            'agent_rate': 1000,
            'physics_rate': 1000,
            'real_time_factor': max_float,
            'simp_model_names': ['monopod_v1'],
            })

register(
    id='Monopod-v1.0.1',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=5000,
    kwargs={'task_cls': tasks.monopod_v1_0_1_balancing.MonopodBalancing,
            'agent_rate': 1000,
            'physics_rate': 1000,
            'real_time_factor': max_float,
            'simp_model_names': ['monopod_v1'],
            })

register(
    id='Monopod-fh-v1.0.0',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=5000,
    kwargs={'task_cls': tasks.monopod_v1_0_0_balancing.MonopodBalancing,
            'agent_rate': 1000,
            'physics_rate': 1000,
            'real_time_factor': max_float,
            'simp_model_names': ['monopod_v1_fh'],
            })
register(
    id='Monopod-fh-v1.0.1',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=5000,
    kwargs={'task_cls': tasks.monopod_v1_0_1_balancing.MonopodBalancing,
            'agent_rate': 1000,
            'physics_rate': 1000,
            'real_time_factor': max_float,
            'simp_model_names': ['monopod_v1_fh'],
            })

register(
    id='Monopod-fh-fby-v1.0.0',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=5000,
    kwargs={'task_cls': tasks.monopod_v1_0_0_balancing.MonopodBalancing,
            'agent_rate': 1000,
            'physics_rate': 1000,
            'real_time_factor': max_float,
            'simp_model_names': ['monopod_v1_fh_fby'],
            })

register(
    id='Monopod-fh-fby-v1.0.1',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=5000,
    kwargs={'task_cls': tasks.monopod_v1_0_1_balancing.MonopodBalancing,
            'agent_rate': 1000,
            'physics_rate': 1000,
            'real_time_factor': max_float,
            'simp_model_names': ['monopod_v1_fh_fby'],
            })
