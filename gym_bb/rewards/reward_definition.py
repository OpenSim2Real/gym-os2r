from gym_ignition.utils.typing import Reward, Observation
from abc import abstractmethod


"""
Base Class
"""


class RewardBase():
    """
    Baseclass for a reward. Please follow this convention when making a new
    reward.

    observation_index is a dictionary which gives the index of the observation
    for a specfied joints position or velocity.
    """

    def __init__(self, observation_index: dict):
        self.observation_index = observation_index
        self.supported_task_modes = []
        self._all_reset_types = ['stand', 'ground']
        self._all_task_modes = ['free_hip',
                                'fixed_hip', 'fixed_hip_and_boom_yaw']

    @abstractmethod
    def calculate_reward(self, obs: Observation) -> Reward:
        pass

    @abstractmethod
    def get_reset_type(self):
        pass

    def is_task_supported(self, task_mode: str):
        return task_mode in self.supported_task_modes

    def get_all_reset_types(self):
        return self._all_reset_types

    def get_supported_task_modes(self):
        return self.supported_task_modes


"""
Balancing tasks. Start from standing and stay standing.
"""


class BalancingV1(RewardBase):
    """
    Standing reward. Start from standing and stay standing.
    """

    def __init__(self, observation_index: dict):
        super().__init__(observation_index)
        self.supported_task_modes = self._all_task_modes

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
    Standing reward. Start from ground and stand up.
    """

    def __init__(self, observation_index: dict):
        super().__init__(observation_index)
        self.supported_task_modes = self._all_task_modes

    def calculate_reward(self, obs: Observation) -> Reward:
        bp = obs[self.observation_index['planarizer_02_joint_pos']]
        return bp

    def get_reset_type(self):
        return 'ground'


"""
Hopping tasks. Start either standing or from ground. favour circular movement.
"""
