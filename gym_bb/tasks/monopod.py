import abc
import gym
import numpy as np
from typing import Tuple
from gym_ignition.base import task
from gym_ignition.utils.typing import Action, Reward, Observation
from gym_ignition.utils.typing import ActionSpace, ObservationSpace
from scenario import core as scenario
import warnings
from gym_bb.config.config import SettingsConfig
from gym_ignition.utils import logger
from collections import deque


class MonopodTask(task.Task, abc.ABC):

    """
    Monopod task defines the main task functionality for the monopod
    environment. Task requires the kwargs; 'task_mode', 'reward_class',
    'reset_positions'.

    Task must be wrapped in a runtime or randomizer class to use with igntion
    or the real robot.

    Attributes:
        task_mode (str): The defined monopod task. current default tasks;
        'free_hip', 'fixed_hip', 'fixed_hip_and_boom_yaw'.
        reward_class (:class:`gym_bb.rewards.rewards.RewardBase`): Class
        defining the reward. Must have same functions as RewardBase.
        reset_positions (str): Reset locations of the task. currently supports;
        'stand', 'half_stand', 'ground', 'lay', 'float'.
        observation_index (dict): dictionry with the joint_name_pos and
        joint_name_vel as keys with values corresponding to its index in the
        observation space.

    """

    def __init__(self, agent_rate: float, **kwargs):
        self.supported_task_modes = ['free_hip',
                                     'fixed_hip', 'fixed_hip_and_boom_yaw']

        required_kwargs = ['task_mode', 'reward_class', 'reset_positions']
        for rkwarg in required_kwargs:
            if rkwarg not in list(kwargs.keys()):
                raise RuntimeError('Missing required kwarg: ' + rkwarg
                                   + '. We require the following kwargs, '
                                   + str(required_kwargs)
                                   + '\n in the MonopodTask class.'
                                   ' (These can be specified in env init)')
        if len(required_kwargs) != len(list(kwargs.keys())):
            warnings.warn('# WARNING: Supplied Kwargs, ' + str(kwargs)
                          + ' Contains more entries than expected. '
                          'Required Kwargs are ' + str(required_kwargs)
                          + '. Could be caused by config object.',
                          SyntaxWarning, stacklevel=2)
        self.__dict__.update(kwargs)
        try:
            self.cfg = kwargs['config']
        except KeyError:
            self.cfg = SettingsConfig()
        supported_reset_pos = list(self.cfg.get_config('/resets').keys())
        if not set(self.reset_positions).issubset(set(supported_reset_pos)):
            raise RuntimeError('One or more of the reset positions provided'
                               ' were not in the supported reset positions. '
                               + str(supported_reset_pos))
        if self.task_mode not in self.supported_task_modes:
            raise RuntimeError(
             'task mode ' + self.task_mode + ' not supported in '
             'monopod environment.')
        try:
            xpath = 'task_modes/' + self.task_mode + '/spaces'
            self.spaces_definition = self.cfg.get_config(xpath)
        except KeyError:
            raise RuntimeError(
                'task mode ' + self.task_mode + ' does not contain spaces '
                'definition in monopod environment config file.')

        # Initialize the Task base class
        task.Task.__init__(self, agent_rate=agent_rate)
        # Name of the monopod model
        self.model_name = None
        # Space for resetting the task
        self.reset_space = None

        # Get names joints
        self.action_names = [*self.spaces_definition['action']]
        self.joint_names = [*self.spaces_definition['observation']]

        # Create dict of index in obs for obs type
        self.observation_index = {}
        for i, joint in enumerate(self.joint_names):
            self.observation_index[joint + '_pos'] = i
            self.observation_index[joint + '_vel'] = i + len(self.joint_names)
        kwargs['observation_index'] = self.observation_index

        # Initialize Reward Class from Kwarg passed in.
        self.reward = self.reward_class(self.observation_index)
        # Verify that the taskmode is compatible with the reward.
        if not self.reward.is_task_supported(self.task_mode):
            raise RuntimeError(self.task_mode
                               + ' task mode not supported by '
                               + str(self.reward) + ' reward class.')

        self.action_history = deque(maxlen=100)
        # Optionally overwrite the above using **kwargs
        self.__dict__.update(kwargs)

    def create_spaces(self) -> Tuple[ActionSpace, ObservationSpace]:
        """
        Constructs observtion and action spaces for monopod task. Spaces
        definition is defined in ../config/default/settings.yaml ...

        Returns:
            ndarray: action space.
            ndarray: observation space.

        """
        # Create the action space
        action_lims = np.array(list(self.spaces_definition['action'].values()))
        observation_lims = np.array(
            list(self.spaces_definition['observation'].values()))

        # Configure action limits
        low = np.array(action_lims[:, 1])
        high = np.array(action_lims[:, 0])
        action_space = gym.spaces.Box(low=low, high=high, dtype=np.float64)

        # Configure reset limits
        a = observation_lims
        low = np.concatenate((a[:, 1], a[:, 3]))
        high = np.concatenate((a[:, 0], a[:, 2]))
        # Configure the reset space - this is used to check if it exists inside
        # the reset space when deciding whether to reset.
        self.reset_space = gym.spaces.Box(low=low, high=high, dtype=np.float64)
        # Configure the observation space
        observation_space = gym.spaces.Box(low=low*1.2, high=high*1.2,
                                           dtype=np.float64)

        return action_space, observation_space

    def set_action(self, action: Action) -> None:
        """
        Set generalized force target for each controlled joint.

        Args:
            action (ndrray): Generalized force target for each
                             controlled joint.
        Raise:
            RuntimeError: Failed to set joints torque target.

        """
        if not self.action_space.contains(action):
            raise RuntimeError(
                "Action Space does not contain the provided action")
        # Store last actions
        self.action_history.append(action)
        # Set the force value
        model = self.world.get_model(self.model_name)

        for joint, value in zip(self.action_names, action.tolist()):
            # Set torque to value given in action
            if not model.get_joint(joint).set_generalized_force_target(value):
                raise RuntimeError(
                    "Failed to set the torque for joint: " + joint)

    def get_observation(self) -> Observation:
        """
        Returns the current observation state of the monopod.

        Returns:
            ndarray: Array of joint positions nad velocities.

        """
        # Get the model
        model = self.world.get_model(self.model_name)

        # Get the new joint positions and velocities
        pos = model.joint_positions(self.joint_names)
        vel = model.joint_velocities(self.joint_names)

        # Create the observation
        observation = Observation(np.array([*pos, *vel]))
        # Return the observation
        return observation

    def is_done(self) -> bool:
        """
        Checks if the current state of the robot is outside of the reset_space.
        logs the reason for the reset as a debug message.

        Returns:
            bool: True for done, False otherwise.

        """
        # Get the observation
        observation = self.get_observation()

        # The environment is done if the observation is outside its space
        done = not self.reset_space.contains(observation)
        if done:
            reason = ~np.logical_and((observation >= self.reset_space.low), (
                observation <= self.reset_space.high))
            msg = ''
            obs_name = np.array(sorted(self.observation_index.keys(),
                                       key=self.observation_index.get))
            for joint, value in zip(obs_name[reason], observation[reason]):
                msg += joint + " caused reset at %.6f, \t " % value
            logger.debug(msg)
        return done

    def reset_task(self) -> None:
        """
        Resets the environment into default state.
        sets the scenario backend into force controller mode
        Sets the max generalized force for each joint.

        """
        if self.model_name not in self.world.model_names():
            raise RuntimeError("Monopod model not found in the world")

        # Get the model
        model = self.world.get_model(self.model_name)

        # Control the monopod in force mode
        for joint_name in self.action_names:
            joint = model.get_joint(joint_name)
            ok = joint.set_control_mode(scenario.JointControlMode_force)
            ok = ok and joint.set_max_generalized_force(
                max(self.action_space.high))
            if not ok:
                raise RuntimeError(
                    "Failed to change the control mode of the Monopod")

    def get_reward(self) -> Reward:
        """
        Returns the reward for the current monopod state.

        Returns:
            bool: True for done, False otherwise.

        """
        obs = self.get_observation()
        return self.calculate_reward(obs, self.action_history[-1])

    def calculate_reward(self, obs: Observation, action: Action) -> Reward:
        """
        Calculates the reward given observation and action. The reward is
        calculated in a provided reward class defined in the tasks kwargs.

        Args:
            obs (np.array): numpy array with the same size task dimensions as
                            observation space.
            action (np.array): numpy array with the same size task dimensions
                            as action space.

        Returns:
            bool: True for done, False otherwise.
        """
        return self.reward.calculate_reward(obs, action)

    def get_state_info(self, obs: Observation, action: Action) -> Tuple[Reward, bool]:
        """
        Returns the reward and is_done for some observation and action space.

        Args:
            obs (np.array): numpy array with the same size task dimensions as
                            observation space.
            action (np.array): numpy array with the same size task dimensions
                            as action space.

        Returns:
            Reward: Rewrd given the state.
            bool: True for done, False otherwise.

        """
        reward = self.calculate_reward(obs, action)
        done = not self.reset_space.contains(obs)
        return reward, done
