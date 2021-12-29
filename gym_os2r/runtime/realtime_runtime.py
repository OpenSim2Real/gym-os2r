# from gym_ignition.base import runtime, task
# from gym_ignition.utils.typing import Action, Done, Info, Observation, State
# from gym_ignition.utils import logger
#
# from scenario import monopod as scenario
# import numpy as np
#
#
# class RealTimeRuntime(runtime.Runtime):
#     """
#     Implementation of :py:class:`~gym_ignition.base.runtime.Runtime` for real-time
#     execution.
#
#     Warning:
#         This class is not yet complete.
#     """
#
#     def __init__(self, task_cls: type, robot_cls: type, agent_rate: float, **kwargs):
#
#         # World attributes
#         self._world = None
#
#         # Build the environment
#         task_object = task_cls(**kwargs)
#
#         assert isinstance(
#             task_object, task.Task
#         ), "'task_cls' object must inherit from Task"
#
#         super().__init__(task=task_object, agent_rate=agent_rate)
#
#         # Initialize the scenario world through property decorator
#         _ = self.world
#
#         # Initialize the spaces
#         self.action_space, self.observation_space = self.task.create_spaces()
#
#         # Store the spaces also in the task
#         self.task.action_space = self.action_space
#         self.task.observation_space = self.observation_space
#
#         raise NotImplementedError
#
#     # =================
#     # Runtime interface
#     # =================
#
#     def timestamp(self) -> float:
#
#         raise NotImplementedError
#
#     # =================
#     # gym.Env interface
#     # =================
#
#     def step(self, action: Action) -> State:
#
#         # Validate action and robot
#         assert self.action_space.contains(action), "%r (%s) invalid" % (
#             action,
#             type(action),
#         )
#
#         # Set the action
#         ok_action = self.task.set_action(action)
#         assert ok_action, "Failed to set the action"
#
#         # TODO: realtime step
#         # This can be used to ensure the loop runs at some consistent loop speed.
#         # Will need to do testing to find what a realalistic control loop hz is
#
#         # Get the observation
#         observation = self.task.get_observation()
#         assert self.observation_space.contains(observation), "%r (%s) invalid" % (
#             observation,
#             type(observation),
#         )
#
#         # Get the reward
#         reward = self.task.get_reward()
#         assert reward, "Failed to get the reward"
#
#         # Check termination
#         done = self.task.is_done()
#
#         return State((observation, reward, Done(done), Info({})))
#
#     def reset(self) -> Observation:
#
#         # Reset the task
#         self.task.reset_task()
#
#         # # TODO: add pause (for manual reset)
#         # Wait for external input before continuing.
#         input("Press Enter when robot is in its reset position...")
#
#         # Get the observation
#         observation = self.task.get_observation()
#         assert isinstance(observation, np.ndarray)
#
#         if not self.observation_space.contains(observation):
#             logger.warn(
#                 "The observation does not belong to the observation space")
#         return Observation(observation)
#
#     def render(self, mode: str = "human", **kwargs) -> None:
#         raise NotImplementedError
#
#     def close(self) -> None:
#         raise NotImplementedError
#
#     @property
#     def world(self) -> scenario.World:
#
#         if self._world is not None:
#             # assert self._world.valid()
#             return self._world
#
#         # Create the world
#         world = scenario.World()
#
#         # TODO: Set joint limits here
#
#         # Set the world in the task
#         self.task.world = world
#
#         # Store the world in runtime
#         self._world = world
#
#         return self._world
