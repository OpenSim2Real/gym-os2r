import numpy
from . import tasks
from . import models
from . import randomizers
from . import monitor
from . import common
from gym.envs.registration import register


max_float = float(numpy.finfo(numpy.float64).max)

register(
    id='Monopod-Gazebo-v1',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=5000,
    kwargs={'task_cls': tasks.monopod_v1_balancing.MonopodV1Balancing,
            'agent_rate': 1000,
            'physics_rate': 1000,
            'real_time_factor': max_float,
            'simp_model_names': ['monopod_v1'],
            })

register(
    id='Monopod-Gazebo-v2',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=5000,
    kwargs={'task_cls': tasks.monopod_v2_balancing.MonopodV2Balancing,
            'agent_rate': 1000,
            'physics_rate': 1000,
            'real_time_factor': max_float,
            'simp_model_names': ['monopod_v1'],
            })

register(
    id='Monopod-Gazebo-fh-v1',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=5000,
    kwargs={'task_cls': tasks.monopod_v1_balancing_fixed_hip.MonopodV1BalancingFixedHip,
            'agent_rate': 1000,
            'physics_rate': 1000,
            'real_time_factor': max_float,
            'simp_model_names': ['monopod_v1_fh'],
            })

register(
    id='Monopod-Gazebo-fh-fby-v1',
    entry_point='gym_ignition.runtimes.gazebo_runtime:GazeboRuntime',
    max_episode_steps=5000,
    kwargs={'task_cls': tasks.monopod_v1_balancing_fixed_hip_and_boom_yaw.MonopodV1BalancingFixedHipAndBoomYaw,
            'agent_rate': 1000,
            'physics_rate': 1000,
            'real_time_factor': max_float,
            'simp_model_names': ['monopod_v1_fh_fby'],
            })
