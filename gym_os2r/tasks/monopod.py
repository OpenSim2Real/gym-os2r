import abc
import gym
import numpy as np
from typing import Tuple, Deque, Dict
from gym_ignition.base import task
from gym_ignition.utils.typing import Action, Reward, Observation
from gym_ignition.utils.typing import ActionSpace, ObservationSpace
from scenario import core as scenario
import warnings
from gym_os2r.models.config import SettingsConfig
from gym_ignition.utils import logger
from collections import deque


class MonopodTask(task.Task, abc.ABC):

    """
    Monopod task defines the main task functionality for the monopod
    environment. Task must be wrapped in a runtime or randomizer class to use
    with igntion or the real robot.

    Attributes:
        Required *kwargs (dict): Task requires the kwargs; 'task_mode',
            'reward_class', 'reset_positions'.
        task_mode (str): The defined monopod task. current default tasks,
            'free_hip', 'fixed_hip', 'fixed', 'old-free_hip', 'old-fixed_hip',
            'old-fixed'.
        reward_class (:class:`gym_os2r.rewards.RewardBase`): Class defining the reward. Must have same
            functions as RewardBase.
        reset_positions ([str]): Reset locations of the task. currently supports,
            'stand', 'half_stand', 'ground', 'lay', 'float'.
        observation_index (dict): dictionry with the joint_name_pos and
            joint_name_vel as keys with values corresponding to its index in
            the observation space.
    """

    def __init__(self, agent_rate: float, **kwargs):
        self.supported_task_modes = ['free_hip', 'fixed_hip', 'fixed',
                                     'fixed_hip_torque', 'simple',
                                     'fixed_hip_simple']

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
        self.model = None

        # Space for resetting the task
        self.reset_space = None
        self.current_reset_orientation = None

        # Get names joints
        self.action_names = [*self.spaces_definition['action']]
        self.joint_names = [*self.spaces_definition['observation']]

        # Create dict of index in obs for obs type
        self.observation_index = {}

        history_len = 10

        self.action_history = deque([np.zeros(len(self.action_names)) for i in range(history_len)],
                                    maxlen=history_len)

        self.observing_measured_torque = self.spaces_definition['observing_measured_torque']
        self.observation_name_mask = self.spaces_definition['observation_mask']
        # Optionally overwrite the above using **kwargs
        self.__dict__.update(kwargs)

    def create_spaces(self) -> Tuple[ActionSpace, ObservationSpace]:
        """
        Constructs observtion and action spaces for monopod task. Spaces
        definition is defined in `../config/default/settings.yaml ...`

        Returns:
            (ndarray): action space.
            (ndarray): observation space.

        """
        # ======= Actions ===============
        # Create the max torques. Dict are ordered in >3.6
        self.max_torques = np.array(
            list(self.spaces_definition['action'].values()))

        # Configure action limits between -1 and 1 which will be scaled by max
        # torque later
        low_act = np.array([-1, -1])
        high_act = np.array([1, 1])
        action_space = gym.spaces.Box(low=low_act, high=high_act, dtype=np.float64)

        # ======= Observations ===============
        obs_list = []
        for joint, joint_info in self.spaces_definition['observation'].items():
            obs_list.append(joint_info['limits'])

        # Configure reset limits assuming no mask or torque observation
        obs_lim = np.array(obs_list)
        low = np.concatenate((obs_lim[:, 1], obs_lim[:, 3]))
        high = np.concatenate((obs_lim[:, 0], obs_lim[:, 2]))

        # If observing_measured_torque then add that to end of obs spaces lims
        if self.observing_measured_torque:
            low = np.array([*low, *low_act])
            high = np.array([*high, *high_act])

        # Create obs index and get low and high spaces after masking out unwanted dims
        masked_i = 0
        self.observation_mask = []
        self.velocities_index = []
        obs_index = {}
        observation_names = [*[n+'_pos' for n in self.joint_names], *[n+'_vel' for n in self.joint_names]]

        # If observing_measured_torque then add name to observed names
        action_names = [n+'_torque' for n in self.action_names]
        observation_names = [*observation_names, *action_names] if self.observing_measured_torque else observation_names

        for obs_i, obs_name in enumerate(observation_names):
            if obs_name not in self.observation_name_mask:
                obs_index[obs_name] = masked_i
                self.observation_mask.append(obs_i)
                if '_vel' in obs_name:
                    self.velocities_index.append(masked_i)
                masked_i += 1

        low = low[self.observation_mask]
        high = high[self.observation_mask]
        self.observation_index = obs_index

        # Store which joint are set and defined as periodic.
        # Force their limit into (-pi, pi)
        self.periodic_joints = []
        for joint, joint_info in self.spaces_definition['observation'].items():
            if joint_info['periodic_pos'] and joint + '_pos' in self.observation_index:
                self.periodic_joints.append(self.observation_index[joint + '_pos'])

        # print('original obs names: ', observation_names)
        # print('obs_index:', self.observation_index)
        # print('obs_mask:', self.observation_mask)
        # print('periodic:', self.periodic_joints)

        low[self.periodic_joints] = -(np.pi+np.finfo(float).eps)
        high[self.periodic_joints] = (np.pi+np.finfo(float).eps)

        #   Store info to normalize the observtion
        self.obs_limits = {'high': high.copy(), 'low': low.copy()}
        self.mask_inf_obs = np.zeros(len(high), dtype=bool)
        self.mask_inf_obs[self.velocities_index] = True
        low[:] = -1
        high[:] = 1

        # define obs space
        obs_space = gym.spaces.Box(low=low, high=high, dtype=np.float64)
        #================== Reward definition =================
                # Initialize reward class with setup observation info
        self.reward = self.reward_class(self.observation_index, normalized=True)
        # Verify that the taskmode is compatible with the reward.
        assert self.reward.is_task_supported(self.task_mode), f'\'{self.task_mode}\' task mode not supported by reward class \'{self.reward}\''


        #  ================ reset spaces ==================
        # Configure the reset space - this is used to check if it exists inside
        # the reset space when deciding whether to reset. Narrow reset space by machine const.
        self.reset_space = gym.spaces.Box(low=low+np.finfo(float).eps, high=high-np.finfo(float).eps, dtype=np.float64)

        return action_space, obs_space

    def set_action(self, action: Action, store_action : bool = True) -> bool:
        """
        Set generalized force target for each controlled joint.

        Args:
            action (ndrray): Generalized force target for each
                             controlled joint.
            store_action (bool): True to store action taken in action history
                             otherwise false to ignore.
        Return:
            (bool): True if success otherwise false.
        Raise:
            (RuntimeError): Failed to set joints torque target.

        """

        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        # Set the force value
        data = self.max_torques * action

        assert self.model.set_joint_generalized_force_targets(
               data, self.action_names
               ), "Failed to set the torque target for joint (%s)" % (
            self.action_names
        )

        # Store last actions
        if store_action:
            self.action_history.appendleft(
                np.array(self.model.joint_generalized_force_targets(
                self.action_names))/self.max_torques)
        return True

    def get_observation(self) -> Observation:
        """
        Returns the current observation state of the monopod.

        Returns:
            (ndarray): Array of joint positions and velocities.

        """

        # Get the new joint positions and velocities
        pos = self.model.joint_positions(self.joint_names)
        vel = self.model.joint_velocities(self.joint_names)

        obs_list = [*pos, *vel]

        if self.observing_measured_torque:
            obs_list = [*obs_list, *self.action_history[1]]

        # Create the observation
        observation = Observation(np.array(obs_list)[self.observation_mask])
        # Set periodic observation --> remainder[(phase + pi)/(2pi)] - pi
        # maps angle --> [-pi, pi)
        observation[self.periodic_joints] = np.mod(
            (observation[self.periodic_joints] + np.pi), (2 * np.pi)) - np.pi

        # Normalize observations
        high = self.obs_limits['high']
        low = self.obs_limits['low']

        # print('obser pre scale: ', obself.position_indexservation)
        m = self.mask_inf_obs
        observation[~m] = 2*(observation[~m] - low[~m])/(high[~m] - low[~m])-1
        observation[m] = np.tanh(0.05 * observation[m])
        # print('obser post scale: ', observation)
        return observation

    def is_done(self) -> bool:
        """
        Checks if the current state of the robot is outside of the reset_space.
        logs the reason for the reset as a debug message.

        Returns:
            (bool): True for done, False otherwise.

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
        Sets the max generalized force for eachcontrolled joint.

        """

        # Control the monopod in force mode
        ok = self.model.set_joint_control_mode(scenario.JointControlMode_force,
                                               self.action_names)

        # set max generalized force for action joints
        ok = ok and all([self.model.get_joint(
            joint_name).set_joint_max_generalized_force(
            [self.action_space.high[i] * self.max_torques[i]]) for i,
            joint_name in enumerate(self.action_names)])

        assert ok, "Failed to change the control mode of the Monopod Model."

    def get_reward(self) -> Reward:
        """
        Returns the reward for the current monopod state.

        Returns:
            (bool): True for done, False otherwise.

        """
        obs = self.get_observation()
        return self.calculate_reward(obs, self.action_history)

    def calculate_reward(self, obs: Observation, action: Deque[Action]) -> Reward:
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
        return self.reward.calculate_reward(obs, action)

    def get_state_info(self, obs: Observation, actions: Deque[Action]) -> Tuple[Reward, bool]:
        """
        Returns the reward and is_done for some observation and action space.

        Args:
            obs (np.array): numpy array with the same size task dimensions as
                            observation space.
            actions Deque[np.array]: Deque of actions taken by the environment
                            numpy array with the same size task dimensions
                            as action space.

        Returns:
            (Reward): Rewrd given the state.
            (bool): True for done, False otherwise.

        """
        reward = self.calculate_reward(obs, action)
        done = not self.reset_space.contains(obs)
        return reward, done

    def get_info(self) -> Dict:
        """
        Return the info dictionary.
        Returns:
            A ``dict`` with extra information of the task.
        """
        return {'reset_orientation': self.current_reset_orientation}
