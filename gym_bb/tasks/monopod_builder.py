import numpy as np
from .monopod_base import MonopodBase
from .rewards.reward_definition import get_reward, supported_rewards
from gym_ignition.utils.typing import Reward, Observation


class MonopodBuilder(MonopodBase):
    """
    Default parameters are used in the init method for torque,
    position/rotation limits, and reset angle. Any of these values can be
    redefined by passing in the corresponding kwargs.
    """

    def __init__(self, agent_rate, **kwargs):
        required_kwargs = ['supported_models', 'task_mode', 'reward_type']
        for rkwarg in required_kwargs:
            if rkwarg not in list(kwargs.keys()):
                raise RuntimeError('Missing required kwarg: ' + rkwarg
                                   + '. We require the following kwargs, '
                                   + str(required_kwargs)
                                   + '\n in the MonopodBuilder class.'
                                   ' (These can be specified in env init)')

        task_mode = kwargs['task_mode']
        reward_type = kwargs['reward_type']
        if reward_type not in supported_rewards().keys():
            raise RuntimeError(
                reward_type
                + ' Reward not found in supported reward definitions.')
        if task_mode not in supported_rewards()[reward_type]:
            raise RuntimeError(task_mode
                               + ' task mode not supported by '
                               + reward_type + ' reward type.')

        self.spaces_definition = {}
        obs_space = self.obs_factory(task_mode)
        self.spaces_definition['observation'] = obs_space()
        super().__init__(agent_rate, **kwargs)

    def calculate_reward(self, obs: Observation) -> Reward:
        reward_definition = get_reward(self.reward_type)
        return reward_definition(obs)

    def obs_factory(self, task_mode):
        if task_mode == 'free_hip':
            return self._free_hip
        elif task_mode == 'fixed_hip':
            return self._fixed_hip
        elif task_mode == 'fixed_hip_and_boom_yaw':
            return self._fixed_hip_and_boom_yaw
        else:
            raise RuntimeError(
                'task mode ' + task_mode + ' not supported in '
                'monopod environment.')

    def _free_hip(self):
        return {
            # pos high, pos low, vel high, vel low
            'upper_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            'lower_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            "planarizer_02_joint": [1.57, -1.57, np.inf, -np.inf],
            "planarizer_01_joint": [np.inf, -np.inf, np.inf, -np.inf],
            'hip_joint': [3.14, -3.14, np.inf, -np.inf]
            }

    def _fixed_hip(self):
        return {
            # pos high, pos low, vel high, vel low
            'upper_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            'lower_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            "planarizer_02_joint": [1.57, -1.57, np.inf, -np.inf],
            "planarizer_01_joint": [np.inf, -np.inf, np.inf, -np.inf]
            }

    def _fixed_hip_and_boom_yaw(self):
        return {
            # pos high, pos low, vel high, vel low
            'upper_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            'lower_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            "planarizer_02_joint": [1.57, -1.57, np.inf, -np.inf]
            }
