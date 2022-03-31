from gym_ignition.utils.typing import Reward, Observation, Action
from abc import abstractmethod
from .rewards_utils import tolerance
import numpy as np
from typing import Deque


# Base Class
class RewardBase():
    """
    Baseclass for rewards. Please follow this convention when making a new
    reward.

    observation_index is a dictionary which gives the index of the observation
    for a specfied joints position or velocity.
    """

    def __init__(self, observation_index: dict, normalized: bool):
        self.observation_index = observation_index
        self.current_reset_obs = None
        self.normalized = normalized
        self.supported_task_modes = []
        self._all_task_modes = ['free_hip', 'fixed_hip', 'fixed', 'simple',
                                'fixed_hip_torque', 'fixed_hip_simple']

    @abstractmethod
    def calculate_reward(self, obs: Observation, actions: Deque[Action]) -> Reward:
        """
        Calculates the reward given observation and action. The reward is
        calculated in a provided reward class defined in the tasks kwargs.

        Args:
            obs (np.array): numpy array with the same size task dimensions as
                            observation space.
            actions Deque[np.array]: Deque of actions taken by the environment
                            numpy array with the same size task dimensions
                            as action space.

        Returns:
            (bool): True for done, False otherwise.
        """
        pass

    def is_task_supported(self, task_mode: str):
        """
        Check if the 'task_mode' is supported by the reward function.

        Args:
            task_mode (str): name of task mode.

        Returns:
            (bool): True for supported, False otherwise.
        """
        return task_mode in self.supported_task_modes

    def get_supported_task_modes(self):
        """
        Get list of tasks supported by the reward function

        Returns:
            (list): list of supported task modes.
        """
        return self.supported_task_modes

    def store_reset_obs(self, obs: Observation):
        """
        Store observation on reset.

        Args:
            obs (np.array): numpy array with the same size task dimensions as
                            observation space.
        """
        self.current_reset_obs = obs


# Balancing tasks
class BalancingV1(RewardBase):
    """
    Balancing reward. Start from standing positions and stay standing.
    """

    def __init__(self, observation_index: dict, normalized: bool):
        super().__init__(observation_index, normalized)
        self.supported_task_modes = ['free_hip', 'fixed_hip', 'fixed_hip_torque', 'fixed_hip_simple', 'fixed']

    def calculate_reward(self, obs: Observation, actions: Deque[Action]) -> Reward:
        action = actions[0]
        _BALANCE_HEIGHT = 0.11/1.57*self.normalized + 0.11*(1-self.normalized)
        bp = obs[self.observation_index['planarizer_pitch_joint_pos']]
        # print(bp)
        balancing = tolerance(bp, (_BALANCE_HEIGHT, 4*_BALANCE_HEIGHT))
        return balancing

class BalancingV2(RewardBase):
    """
    Balancing reward. Start from standing positions and stay standing. Smaller
    control signals are favoured.
    """

    def __init__(self, observation_index: dict, normalized: bool):
        super().__init__(observation_index, normalized)
        self.supported_task_modes = ['free_hip', 'fixed_hip', 'fixed_hip_torque', 'fixed_hip_simple', 'fixed']

    def calculate_reward(self, obs: Observation, actions: Deque[Action]) -> Reward:
        action = actions[0]
        _BALANCE_HEIGHT = 0.11/1.57*self.normalized + 0.11*(1-self.normalized)
        bp = obs[self.observation_index['planarizer_pitch_joint_pos']]
        balancing = tolerance(bp, (_BALANCE_HEIGHT, 4*_BALANCE_HEIGHT))
        small_control = tolerance(action,
                                  margin = 1, value_at_margin = 0.4,
                                  sigmoid = 'quadratic')
        return balancing * np.prod(small_control)

class BalancingV3(RewardBase):
    """
    Balancing reward. Start from standing positions and stay standing.
    Small changes in control signal magnitude are favoured.
    """

    def __init__(self, observation_index: dict, normalized: bool):
        super().__init__(observation_index, normalized)
        self.supported_task_modes = ['free_hip', 'fixed_hip', 'fixed_hip_torque', 'fixed_hip_simple', 'fixed']

    def calculate_reward(self, obs: Observation, actions: Deque[Action]) -> Reward:
        action = actions[0]
        action_old = actions[1]
        _BALANCE_HEIGHT = 0.11/1.57*self.normalized + 0.11*(1-self.normalized)
        bp = obs[self.observation_index['planarizer_pitch_joint_pos']]
        balancing = tolerance(bp, (_BALANCE_HEIGHT, 4*_BALANCE_HEIGHT), margin=0.01, sigmoid='long_tail')
        # small_control = tolerance(action,
        #                           margin = 1, value_at_margin = 0.1,
        #                           sigmoid = 'quadratic')
        # return balancing * np.prod(small_control)
        small_delta_control = tolerance(action-action_old,
                                  margin = 1, value_at_margin = 0.1,
                                  sigmoid = 'quadratic')

        # small_delta_control = tolerance(action-action_old,
        #                           margin = 0.2, value_at_margin = 0.1,
        #                           sigmoid = 'gaussian')

        return balancing * np.prod(small_delta_control)


class BalancingV4(RewardBase):
    """
    Same as BalancingV3 except has torque delta cost
    """

    def __init__(self, observation_index: dict, normalized: bool):
        super().__init__(observation_index, normalized)
        self.supported_task_modes = self._all_task_modes

    def calculate_reward(self, obs: Observation, actions: Deque[Action]) -> Reward:
        _BALANCE_HEIGHT = 0.15/1.57*self.normalized + 0.15*(1-self.normalized)
        print(self.current_reset_obs)
        # How about the below...like a normal person
        # _BALANCE_HEIGHT = 0.15/1.57 if self.normalized else 0.15

        bp = obs[self.observation_index['planarizer_pitch_joint_pos']]
        by = obs[self.observation_index['planarizer_yaw_joint_pos']]
        # print(bp)
        balancing_reward = tolerance(bp, (_BALANCE_HEIGHT, np.inf), 
                                    sigmoid='gaussian', margin=0.08) # 0 or 1
        torq_deltas = abs(actions[0] - actions[1])
        delta_cost = np.mean(torq_deltas) / 2
        action_cost = abs(actions[0]).sum() / 2 # in [0,1]
        offset_cost = abs(by) 
        # return (1 - action_cost) * balancing_reward * (1 - delta_cost)
        return (1 - action_cost) * balancing_reward * (1 - offset_cost) * (1 - delta_cost)


# Standing tasks

# class StandingV1(RewardBase):
#     """
#     Standing reward. Start from ground and stand up.
#     """

#     def __init__(self, observation_index: dict, normalized: bool):
#         super().__init__(observation_index, normalized)
#         self.supported_task_modes = ['free_hip', 'fixed_hip', 'fixed_hip_torque', 'fixed_hip_simple', 'fixed']

#     def calculate_reward(self, obs: Observation, actions: Deque[Action]) -> Reward:
#         action = actions[0]
#         _STAND_HEIGHT = 0.11/1.57*self.normalized + 0.11*(1-self.normalized)
#         bp = obs[self.observation_index['planarizer_pitch_joint_pos']]
#         standing = tolerance(bp, (_STAND_HEIGHT, 4*_BALANCE_HEIGHT))
#         return standing

class HoppingV1(RewardBase):
    """
    Balancing reward. Start from standing positions and stay standing. Smaller
    control signals are favoured.
    """

    def __init__(self, observation_index: dict, normalized: bool):
        super().__init__(observation_index, normalized)
        self.supported_task_modes = ['free_hip', 'fixed_hip', 'fixed_hip_torque', 'fixed_hip_simple', 'fixed']

    def calculate_reward(self, obs: Observation, actions: Deque[Action]) -> Reward:
        action = actions[0]
        action_old = actions[1]
        _BALANCE_HEIGHT = 0.11/1.57*self.normalized + 0.11*(1-self.normalized)
        bp = obs[self.observation_index['planarizer_pitch_joint_pos']]
        # balancing = tolerance(bp, (_BALANCE_HEIGHT, 4*_BALANCE_HEIGHT), margin=0.01, sigmoid='long_tail')
        balancing = tolerance(bp, (_BALANCE_HEIGHT, 4*_BALANCE_HEIGHT))
        # small_control = tolerance(action,
        #                           margin = 1, value_at_margin = 0.1,
        #                           sigmoid = 'quadratic')

        small_delta_control = tolerance(action-action_old,
                                  margin = 0.1, value_at_margin = 0,
                                  sigmoid = 'quadratic')
        h_vel = obs[self.observation_index['planarizer_yaw_joint_vel']]
        move = tolerance(h_vel,bounds=(0.25, 0.3),
                                margin=0.15, value_at_margin=0.1,
                                sigmoid='tanh_squared')

        return balancing * np.prod(small_delta_control) * move
        # return balancing * np.prod(small_delta_control) * np.prod(small_control)

class StraightV1(RewardBase):
    """
    Standing reward. Start from ground and stand up.
    """

    def __init__(self, observation_index: dict, normalized: bool):
        super().__init__(observation_index, normalized)
        self.supported_task_modes = ['simple']#self._all_task_modes

    def calculate_reward(self, obs: Observation, actions: Deque[Action]) -> Reward:
        action = actions[0]

        small_control = tolerance(action/20, margin=1,
                                      value_at_margin=0,
                                      sigmoid='quadratic').mean()
        small_control = (4 + small_control) / 5

        hip = obs[self.observation_index['hip_joint_pos']]
        knee = obs[self.observation_index['knee_joint_pos']]

        hip_reward = tolerance(hip, bounds=(0, 0), margin=1,sigmoid='linear')

        knee_reward = tolerance(knee, bounds=(0, 0), margin=1,sigmoid='linear')

        return hip_reward*knee_reward*small_control
