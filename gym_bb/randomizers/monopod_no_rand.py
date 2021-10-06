from typing import Union
import random
from gym_bb import tasks
from gym_bb.models import monopod
from gym_ignition.randomizers import gazebo_env_randomizer
from gym_ignition.randomizers.gazebo_env_randomizer import MakeEnvCallable
from gym_ignition.utils.typing import Observation
from gym_bb.rewards.rewards import RewardBase

# Tasks that are supported by this randomizer. Used for type hinting.
SupportedTasks = Union[tasks.monopod_builder.MonopodBuilder]


class MonopodEnvNoRandomizer(gazebo_env_randomizer.GazeboEnvRandomizer):
    """
    Dummy environment randomizer for monopod tasks.

    Check :py:class:`gym_bb.randomizers.monopod.MonopodRandomizersMixin`
    for an example that randomizes the task, the physics, and the model.
    """

    def __init__(self, env: MakeEnvCallable, reward_class: RewardBase, **kwargs):

        super().__init__(env=env, reward_class=reward_class, **kwargs)

    def randomize_task(self, task: SupportedTasks, **kwargs) -> None:
        """
        Prepare the scene for monopod tasks. It simply removes the monopod of
        the previous rollout and inserts a new one in the default state. Then,
        the active Task will reset the state of the monopod depending on the
        implemented decision-making logic.
        """

        if "gazebo" not in kwargs:
            raise ValueError("gazebo kwarg not passed to the task randomizer")

        gazebo = kwargs["gazebo"]

        # Remove the model from the simulation
        if task.model_name is not None and task.model_name in task.world.model_names():
            if not task.world.to_gazebo().remove_model(task.model_name):
                raise RuntimeError(
                    "Failed to remove the monopod from the world")
        # Execute a paused run to process model removal
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

        # Check env supports at least one model
        if not len(task.supported_models):
            raise RuntimeError('No monopod models support by environement...')
        # Insert a new monopod model (randomally choosen a compatible one)
        model = monopod.Monopod(world=task.world,
                                monopod_version=random.choice(
                                    task.supported_models))

        # Store the model name in the task
        task.model_name = model.name()

        # TODO: Make the reset position adjust to the Reward method.
        pos_reset = vel_reset = [0]*len(task.joint_names)
        pos_reset[task.joint_names.index(
            'boom_pitch_joint')] = 0.3

        g_model = model.to_gazebo()
        ok_pos = g_model.reset_joint_positions(pos_reset, task.joint_names)
        ok_vel = g_model.reset_joint_velocities(vel_reset, task.joint_names)

        if not (ok_pos and ok_vel):
            raise RuntimeError("Failed to reset the monopod state")

        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def get_state_info(self, state: Observation):
        return self.env.unwrapped.task.get_state_info(state)
