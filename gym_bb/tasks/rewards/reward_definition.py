from gym_ignition.utils.typing import Reward, Observation
from abc import abstractmethod
import sys

_all_reset_types = ['stand', 'ground']
_all_task_modes = ['free_hip', 'fixed_hip', 'fixed_hip_and_boom_yaw']
Supported_rewards = {
    'StandingV1': _all_task_modes,
    'BalancingV1': _all_task_modes
}

"""
Public Methods
"""


def get_reward_class(reward_class_name: str):
    return str_to_class(reward_class_name)


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def supported_rewards():
    """
    Returns dictionary where each key is a supported reward method and
    each task that the reward supports.
    """
    return Supported_rewards


"""
Base Class
"""


class RewardBase():
    """
    Baseclass for a reward. Please follow this convention when making a reward.
    """

    def __init__(self, observation_index: dict):
        self.observation_index = observation_index

    @abstractmethod
    def calculate_reward(self, obs: Observation) -> Reward:
        pass

    @abstractmethod
    def get_reset_type(self):
        pass


"""
Balancing tasks. Start from standing and stay standing.
"""


class BalancingV1(RewardBase):
    """
    Standing reward
    """

    def calculate_reward(self, obs: Observation) -> Reward:
        bp = obs[self.observation_index['planarizer_02_joint_pos']]
        return bp

    def get_reset_type(self):
        return 'stand'


"""
Standing tasks. Start from ground and stand up.
"""


class StandingV1(RewardBase):
    """
    Standing reward
    """

    def calculate_reward(self, obs: Observation) -> Reward:
        bp = obs[self.observation_index['planarizer_02_joint_pos']]
        return bp

    def get_reset_type(self):
        return 'ground'


"""
Hopping tasks. Start either standing or from ground. favour circular movement.
"""
