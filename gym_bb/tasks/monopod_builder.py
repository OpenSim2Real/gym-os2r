import numpy as np
from .monopod_base import MonopodBase
from .rewards import calculate_balancing_rewards
from gym_ignition.utils.typing import Reward, Observation


class MonopodBuilder(MonopodBase):
    """
    Default parameters are used in the init method for torque, position/rotation
    limits, and reset angle. Any of these values can be redefined by passing in
    the corresponding kwargs.
    """

    def __init__(self, agent_rate, **kwargs):

        self.spaces_definition = {}
        env_setup = {
            # Free hip
            'monopod_v1': self._create_original,
            # Fixed hip
            'monopod_v1_fh': self._create_fh,
            # Fixed Hip and boom yaw
            'monopod_v1_fh_fby': self._create_fh_fby
        }
        try:
            env_setup[kwargs['supported_models'][0]]()
        except KeyError:
            raise RuntimeError(
                'Model ' + kwargs['supported_models'][0] + 'not supported in'
                'monopod environment')

        super().__init__(agent_rate, **kwargs)

    def calculate_reward(self, obs: Observation) -> Reward:
        reward_setup = {
            'Standing_v1': calculate_balancing_rewards.standing_v1,
            'Balancing_v1': calculate_balancing_rewards.balancing_v1,
        }
        return reward_setup[self.reward_type](obs)

    def _create_original(self):
        self.spaces_definition['observation'] = {
            # pos high, pos low, vel high, vel low
            'upper_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            'lower_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            "planarizer_02_joint": [1.57, -1.57, np.inf, -np.inf],
            "planarizer_01_joint": [np.inf, -np.inf, np.inf, -np.inf],
            'hip_joint': [3.14, -3.14, np.inf, -np.inf]
            }

    def _create_fh(self):
        self.spaces_definition['observation'] = {
            # pos high, pos low, vel high, vel low
            'upper_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            'lower_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            "planarizer_02_joint": [1.57, -1.57, np.inf, -np.inf],
            "planarizer_01_joint": [np.inf, -np.inf, np.inf, -np.inf]
            }

    def _create_fh_fby(self):
        self.spaces_definition['observation'] = {
            # pos high, pos low, vel high, vel low
            'upper_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            'lower_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            "planarizer_02_joint": [1.57, -1.57, np.inf, -np.inf]
            }
