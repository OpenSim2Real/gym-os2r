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

    def __init__(self, observation_index: dict):
        self.observation_index = observation_index
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


# Balancing tasks
class BalancingV1(RewardBase):
    """
    Balancing reward. Start from standing positions and stay standing.
    """

    def __init__(self, observation_index: dict):
        super().__init__(observation_index)
        self.supported_task_modes = ['free_hip', 'fixed_hip', 'fixed_hip_torque', 'fixed_hip_simple', 'fixed']

    def calculate_reward(self, obs: Observation, actions: Deque[Action]) -> Reward:
        action = actions[0]
        _BALANCE_HEIGHT = 0.11/1.57
        bp = obs[self.observation_index['planarizer_pitch_joint_pos']]
        # print(bp)
        balancing = tolerance(bp, (_BALANCE_HEIGHT, 4*_BALANCE_HEIGHT))
        return balancing

class BalancingV2(RewardBase):
    """
    Balancing reward. Start from standing positions and stay standing. Smaller
    control signals are favoured.
    """

    def __init__(self, observation_index: dict):
        super().__init__(observation_index)
        self.supported_task_modes = ['free_hip', 'fixed_hip', 'fixed_hip_torque', 'fixed_hip_simple', 'fixed']

    def calculate_reward(self, obs: Observation, actions: Deque[Action]) -> Reward:
        action = actions[0]
        _BALANCE_HEIGHT = 0.11/1.57
        bp = obs[self.observation_index['planarizer_pitch_joint_pos']]
        balancing = tolerance(bp, (_BALANCE_HEIGHT, 4*_BALANCE_HEIGHT))
        small_control = tolerance(action,
                                  margin=1, value_at_margin=0,
                                  sigmoid='quadratic').mean()
        small_control = (small_control + 4) / 5
        return balancing * small_control

class BalancingV3(RewardBase):
    """
    Balancing reward. Start from standing positions and stay standing. Smaller
    control signals are favoured.
    """

    def __init__(self, observation_index: dict):
        super().__init__(observation_index)
        self.supported_task_modes = self._all_task_modes

    def calculate_reward(self, obs: Observation, actions: Deque[Action]) -> Reward:
        _BALANCE_HEIGHT = 0.1
        bp = obs[self.observation_index['planarizer_pitch_joint_pos']]
        by = obs[self.observation_index['planarizer_yaw_joint_pos']]
        # print(actions)
        # print(f"bp: {bp} \n action: {action} \n")
        # print(obs[self.observation_index['planarizer_yaw_joint_vel']])
        balancing_reward = tolerance(bp, (_BALANCE_HEIGHT, float("inf"))) # 0 or 1
        action_cost = abs(actions[0]).sum() / 40 # Divide by 40 to be in [0,1]
        offset_cost = abs(by) 
        return (1 - action_cost) * balancing_reward * (1 - offset_cost)

# class BalancingV3(RewardBase):
#     """
#     Balancing reward. Start from standing positions and stay standing. Smaller
#     control signals are favoured.
#     """

#     def __init__(self, observation_index: dict):
#         super().__init__(observation_index)
#         self.supported_task_modes = ['free_hip', 'fixed_hip', 'fixed_hip_torque', 'fixed']

#     def calculate_reward(self, obs: Observation, actions: Deque[Action]) -> Reward:
#         action = actions[0]
#         action_old = actions[1]
#         print(actions)
#         print("\n")
#         _BALANCE_HEIGHT = 0.1
#         bp = obs[self.observation_index['planarizer_pitch_joint_pos']]
#         balancing = tolerance(bp, (_BALANCE_HEIGHT, 0.4))
#         small_control = tolerance(action,
#                                   margin = 1, value_at_margin = 0.33,
#                                   sigmoid = 'quadratic')
#         # small_delta_control = tolerance(action-action_old,
#         #                           margin = 1, value_at_margin = 0,
#         #                           sigmoid = 'quadratic')
#         # return balancing * np.prod(small_control) * np.prod(small_delta_control)
#         return balancing * np.prod(small_control)

# Standing tasks

class StandingV1(RewardBase):
    """
    Standing reward. Start from ground and stand up.
    """

    def __init__(self, observation_index: dict):
        super().__init__(observation_index)
        self.supported_task_modes = ['free_hip', 'fixed_hip', 'fixed_hip_torque', 'fixed_hip_simple', 'fixed']

    def calculate_reward(self, obs: Observation, actions: Deque[Action]) -> Reward:
        action = actions[0]
        _STAND_HEIGHT = 0.2/1.57
        bp = obs[self.observation_index['planarizer_pitch_joint_pos']]
        standing = tolerance(bp, (_STAND_HEIGHT, 4*_BALANCE_HEIGHT))
        return standing

class StandingV2(RewardBase):
    """
    Standing reward. Start from ground and stand up.
    """

    def __init__(self, observation_index: dict):
        super().__init__(observation_index)
        self.supported_task_modes = ['free_hip', 'fixed_hip', 'fixed_hip_torque', 'fixed_hip_simple', 'fixed']

    def calculate_reward(self, obs: Observation, actions: Deque[Action]) -> Reward:
        action = actions[0]
        _STAND_HEIGHT = 0.1/1.57
        bp = obs[self.observation_index['planarizer_pitch_joint_pos']]*50
        #standing = tolerance(bp, (_STAND_HEIGHT, 0.15))
        #TODO Fix hardcoded normalized action
        action_cost = 0.1 * np.square(action/20).sum()
        return bp-action_cost + 1

class StandingV3(RewardBase):
    """
    Standing reward. Start from ground and stand up.
    """

    def __init__(self, observation_index: dict):
        super().__init__(observation_index)
        self.supported_task_modes = ['free_hip', 'fixed_hip', 'fixed_hip_torque', 'fixed_hip_simple', 'fixed']

    def calculate_reward(self, obs: Observation, actions: Deque[Action]) -> Reward:
        action = actions[0]
        _STAND_HEIGHT = 0.1/1.57
        _IDEAL_ANGLE = 0.5
        _ANGLE_LIMIT = 6.3

        standing = tolerance(obs[self.observation_index['planarizer_pitch_joint_pos']],
                                 bounds=(_STAND_HEIGHT, float('inf')),
                                 margin=_STAND_HEIGHT/4)
        # knee_reward = tolerance(obs[self.observation_index['knee_joint_pos']],
        #                          bounds=(-_IDEAL_ANGLE, _IDEAL_ANGLE),
        #                          margin=_IDEAL_ANGLE/4)
        knee_reward = tolerance(obs[self.observation_index['knee_joint_pos']],
                                 bounds=(-_IDEAL_ANGLE, _IDEAL_ANGLE),
                                 margin=0)

        hip_within_limit = tolerance(obs[self.observation_index['hip_joint_pos']],
                                 bounds=(-_ANGLE_LIMIT, _ANGLE_LIMIT),
                                 margin=0)

        boom_connector_within_limit = tolerance(obs[self.observation_index['boom_connector_joint_pos']],
                                 bounds=(-_ANGLE_LIMIT, _ANGLE_LIMIT),
                                 margin=0)

        boundaries_reward = hip_within_limit*boom_connector_within_limit

        stand_reward = standing
        small_control = tolerance(action/20, margin=1,
                                      value_at_margin=0,
                                      sigmoid='quadratic').mean()
        small_control = (4 + small_control) / 5

        horizontal_velocity = obs[self.observation_index['planarizer_yaw_joint_vel']]
        dont_move = tolerance(horizontal_velocity, margin=2)
        return round(boundaries_reward*small_control * stand_reward * dont_move*knee_reward,3)

class HoppingV1(RewardBase):
    """
    Balancing reward. Start from standing positions and stay standing. Smaller
    control signals are favoured.
    """

    def __init__(self, observation_index: dict):
        super().__init__(observation_index)
        self.supported_task_modes = ['free_hip', 'fixed_hip', 'fixed_hip_torque', 'fixed_hip_simple', 'fixed']

    def calculate_reward(self, obs: Observation, actions: Deque[Action]) -> Reward:
        action = actions[0]
        action_old = actions[1]
        _BALANCE_HEIGHT = 0.11/1.57
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

    def __init__(self, observation_index: dict):
        super().__init__(observation_index)
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


# Walking tasks


class WalkingV1(RewardBase):
    """
    Walking reward. Start from standing position and attempt to move
    forward while maintaining the standing height and position.
    """

    def __init__(self, observation_index: dict):
        super().__init__(observation_index)
        self.supported_task_modes = ['free_hip', 'fixed_hip', 'fixed_hip_torque', 'fixed_hip_simple']

    def calculate_reward(self, obs: Observation, actions: Deque[Action]) -> Reward:
        action = actions[0]
        _STAND_HEIGHT = 0.2/1.57
        _HOP_SPEED = 1
        bp_pos = obs[self.observation_index['planarizer_pitch_joint_pos']]
        standing = tolerance(bp_pos, (_STAND_HEIGHT, 4*_BALANCE_HEIGHT))
        by_vel = obs[self.observation_index['planarizer_yaw_joint_vel']]
        hopping = tolerance(by_vel,
                            bounds=(_HOP_SPEED, float('inf')),
                            margin=_HOP_SPEED/2,
                            value_at_margin=0.5,
                            sigmoid='linear')
        return standing * hopping


# Hopping task
class HoppingV1(RewardBase):
    """
    Hopping vertically. Pogo stick-ing
    """

    def __init__(self, observation_index: dict):
        super().__init__(observation_index)
        self.supported_task_modes = self._all_task_modes

    def calculate_reward(self, obs: Observation, action: Action) -> Reward:
        _STAND_HEIGHT = 0.15
        bp = obs[self.observation_index['planarizer_pitch_joint_pos']]
        print(bp, action)
        by = obs[self.observation_index['planarizer_yaw_joint_pos']]
        # # hip_pos = abs(obs[self.observation_index['hip_joint_pos']])
        # # knee_pos = abs(obs[self.observation_index['knee_joint_pos']])

        hopping_reward = tolerance(bp, (_STAND_HEIGHT, np.inf)) # 0 or 1
        hopping_reward *= (bp - _STAND_HEIGHT) / (1.57 - _STAND_HEIGHT)

        # if hopping_reward > 0:
        #     hopping_reward = (bp - _STAND_HEIGHT) / (1.57 - _STAND_HEIGHT) # in [0,1]

        # # # Favour leg being vertical using average distance 
        # # upright_cost = (hip_pos / 6.30 + knee_pos) / 2

        hopping_reward = tolerance(bp, (_STAND_HEIGHT, np.inf)) # 0 or 1
        hopping_reward *= bp / 1.57 # 0 or in [1,2]
        action_cost = abs(action).sum() / 40 # in [0,1]
        offset_cost = abs(by) 
        return hopping_reward * (1 - action_cost) * (1 - offset_cost) # in [0,2]

 
