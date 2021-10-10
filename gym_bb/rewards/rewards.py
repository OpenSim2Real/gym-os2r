from gym_ignition.utils.typing import Reward, Observation, Action
from abc import abstractmethod
from .rewards_utils import tolerance


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
        self._all_task_modes = ['free_hip',
                                'fixed_hip', 'fixed_hip_and_boom_yaw']

    @abstractmethod
    def calculate_reward(self, obs: Observation, action: Action) -> Reward:
        pass

    def is_task_supported(self, task_mode: str):
        return task_mode in self.supported_task_modes

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

    def calculate_reward(self, obs: Observation, action: Action) -> Reward:
        _BALANCE_HEIGHT = 0.2
        bp = obs[self.observation_index['boom_pitch_joint_pos']]
        balancing = tolerance(bp, (_BALANCE_HEIGHT, 0.4))
        return balancing


class BalancingV2(RewardBase):
    """
    Standing reward. Start from standing and stay standing.
    """

    def __init__(self, observation_index: dict):
        super().__init__(observation_index)
        self.supported_task_modes = self._all_task_modes

    def calculate_reward(self, obs: Observation, action: Action) -> Reward:
        _BALANCE_HEIGHT = 0.2
        bp = obs[self.observation_index['boom_pitch_joint_pos']]
        balancing = tolerance(bp, (_BALANCE_HEIGHT, 0.4))
        small_control = tolerance(action,
                                  margin=1, value_at_margin=0,
                                  sigmoid='quadratic').mean()
        small_control = (small_control + 4) / 5
        return balancing * small_control


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

    def calculate_reward(self, obs: Observation, action: Action) -> Reward:
        _STAND_HEIGHT = 0.2
        bp = obs[self.observation_index['boom_pitch_joint_pos']]
        standing = tolerance(bp, (_STAND_HEIGHT, 0.4))
        return standing


"""
Walking tasks. Start from ground and stand up.
"""


class WalkingV1(RewardBase):
    """
    Standing reward. Start from ground and stand up.
    """

    def __init__(self, observation_index: dict):
        super().__init__(observation_index)
        self.supported_task_modes = ['free_hip', 'fixed_hip']

    def calculate_reward(self, obs: Observation, action: Action) -> Reward:
        _STAND_HEIGHT = 0.2
        _HOP_SPEED = 1
        bp_pos = obs[self.observation_index['boom_pitch_joint_pos']]
        standing = tolerance(bp_pos, (_STAND_HEIGHT, 0.4))
        by_vel = obs[self.observation_index['boom_yaw_joint_vel']]
        hopping = tolerance(by_vel,
                            bounds=(_HOP_SPEED, float('inf')),
                            margin=_HOP_SPEED/2,
                            value_at_margin=0.5,
                            sigmoid='linear')
        return standing * hopping


"""
Hopping tasks. Start either standing or from ground. favour circular movement.
"""
