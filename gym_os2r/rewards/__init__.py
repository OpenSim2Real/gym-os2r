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
                                'fixed_hip_torque']

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
        self.supported_task_modes = ['free_hip', 'fixed_hip', 'fixed_hip_torque', 'fixed']

    def calculate_reward(self, obs: Observation, actions: Deque[Action]) -> Reward:
        action = actions[0]
        _BALANCE_HEIGHT = 0.1
        bp = obs[self.observation_index['planarizer_pitch_joint_pos']]
        balancing = tolerance(bp, (_BALANCE_HEIGHT, 0.15))
        return balancing

class BalancingV2(RewardBase):
    """
    Balancing reward. Start from standing positions and stay standing. Smaller
    control signals are favoured.
    """

    def __init__(self, observation_index: dict):
        super().__init__(observation_index)
        self.supported_task_modes = ['free_hip', 'fixed_hip', 'fixed_hip_torque', 'fixed']

    def calculate_reward(self, obs: Observation, actions: Deque[Action]) -> Reward:
        action = actions[0]
        _BALANCE_HEIGHT = 0.2
        bp = obs[self.observation_index['planarizer_pitch_joint_pos']]
        balancing = tolerance(bp, (_BALANCE_HEIGHT, 0.4))
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
        self.supported_task_modes = ['free_hip', 'fixed_hip', 'fixed_hip_torque', 'fixed']

    def calculate_reward(self, obs: Observation, actions: Deque[Action]) -> Reward:
        action = actions[0]
        action_old = actions[1]
        _BALANCE_HEIGHT = 0.1
        bp = obs[self.observation_index['planarizer_pitch_joint_pos']]
        balancing = tolerance(bp, (_BALANCE_HEIGHT, 0.4))
        small_control = tolerance(action,
                                  margin = 1, value_at_margin = 0.2,
                                  sigmoid = 'quadratic')
        # small_delta_control = tolerance(action-action_old,
        #                           margin = 1, value_at_margin = 0,
        #                           sigmoid = 'quadratic')
        # return balancing * np.prod(small_delta_control)
        return balancing * np.prod(small_control)

# Standing tasks

class StandingV1(RewardBase):
    """
    Standing reward. Start from ground and stand up.
    """

    def __init__(self, observation_index: dict):
        super().__init__(observation_index)
        self.supported_task_modes = ['free_hip', 'fixed_hip', 'fixed_hip_torque', 'fixed']

    def calculate_reward(self, obs: Observation, actions: Deque[Action]) -> Reward:
        action = actions[0]
        _STAND_HEIGHT = 0.2
        bp = obs[self.observation_index['planarizer_pitch_joint_pos']]
        standing = tolerance(bp, (_STAND_HEIGHT, 0.4))
        return standing

class StandingV2(RewardBase):
    """
    Standing reward. Start from ground and stand up.
    """

    def __init__(self, observation_index: dict):
        super().__init__(observation_index)
        self.supported_task_modes = ['free_hip', 'fixed_hip', 'fixed_hip_torque', 'fixed']

    def calculate_reward(self, obs: Observation, actions: Deque[Action]) -> Reward:
        action = actions[0]
        _STAND_HEIGHT = 0.1
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
        self.supported_task_modes = ['free_hip', 'fixed_hip', 'fixed_hip_torque', 'fixed']

    def calculate_reward(self, obs: Observation, actions: Deque[Action]) -> Reward:
        action = actions[0]
        _STAND_HEIGHT = 0.1
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
    Standing reward. Start from ground and stand up.
    """

    def __init__(self, observation_index: dict):
        super().__init__(observation_index)
        self.supported_task_modes = ['free_hip', 'fixed_hip', 'fixed_hip_torque', 'fixed']

    def calculate_reward(self, obs: Observation, actions: Deque[Action]) -> Reward:
        action = actions[0]
        _STAND_HEIGHT = 0.12
        _HOP_SPEED = 0.01
        _IDEAL_ANGLE = 0.45
        _MAX_VERTICAL_SPEED = 0.08

        standing = tolerance(obs[self.observation_index['planarizer_pitch_joint_pos']],
                                 bounds=(_STAND_HEIGHT, float('inf')),
                                 margin=_STAND_HEIGHT/4)

        upright = tolerance(obs[self.observation_index['knee_joint_pos']],
                                    bounds=(-_IDEAL_ANGLE, _IDEAL_ANGLE), sigmoid='linear',
                                    margin=0.8, value_at_margin=0)

        stand_reward = standing*upright
        # small_control = tolerance(action/20, margin=1,
        #                               value_at_margin=0,
        #                               sigmoid='quadratic').mean()
        # small_control = (4 + small_control) / 5

        impact_cost = tolerance(obs[self.observation_index['planarizer_pitch_joint_vel']],
                            bounds=(-_MAX_VERTICAL_SPEED, _MAX_VERTICAL_SPEED),
                            margin=1, value_at_margin=0, sigmoid='quadratic')


        horizontal_velocity = obs[self.observation_index['planarizer_yaw_joint_vel']]
        #print("Upward Position")
        #print(obs[self.observation_index['planarizer_pitch_joint_pos']])
        # print("Upward Velocity")
        # print(obs[self.observation_index['planarizer_pitch_joint_vel']])
        # print("KNEE POSITION")
        # print(obs[self.observation_index['knee_joint_pos']])
        move = tolerance(horizontal_velocity,
                        bounds=(_HOP_SPEED, float('inf')),
                        margin=_HOP_SPEED, value_at_margin=0,
                        sigmoid='linear')
        move = (5*move + 1) / 6
        return round( impact_cost*stand_reward * move,3)

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
        self.supported_task_modes = ['free_hip', 'fixed_hip', 'fixed_hip_torque']

    def calculate_reward(self, obs: Observation, actions: Deque[Action]) -> Reward:
        action = actions[0]
        _STAND_HEIGHT = 0.2
        _HOP_SPEED = 1
        bp_pos = obs[self.observation_index['planarizer_pitch_joint_pos']]
        standing = tolerance(bp_pos, (_STAND_HEIGHT, 0.4))
        by_vel = obs[self.observation_index['planarizer_yaw_joint_vel']]
        hopping = tolerance(by_vel,
                            bounds=(_HOP_SPEED, float('inf')),
                            margin=_HOP_SPEED/2,
                            value_at_margin=0.5,
                            sigmoid='linear')
        return standing * hopping
