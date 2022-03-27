from typing import Union, Deque
from gym_os2r import tasks
import numpy as np
from gym_os2r.models import monopod
from gym_ignition.randomizers import gazebo_env_randomizer
from gym_ignition.randomizers.gazebo_env_randomizer import MakeEnvCallable
from gym_ignition.utils.typing import Observation, Action
from gym_os2r.utils.reset import leg_joint_angles

# Tasks that are supported by this randomizer. Used for type hinting.
SupportedTasks = Union[tasks.monopod.MonopodTask]


class MonopodEnvNoRandomizer(gazebo_env_randomizer.GazeboEnvRandomizer):
    """
    Dummy environment randomizer for monopod tasks.

    Check :py:class:`gym_os2r.randomizers.monopod.MonopodRandomizersMixin`
    for an example that randomizes the task, the physics, and the model.
    """

    def __init__(self, env: MakeEnvCallable, **kwargs):

        super().__init__(env=env, **kwargs)

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

        xpath = 'task_modes/' + task.task_mode + '/model'
        monopod_model = task.cfg.get_config(xpath)
        # Insert a new monopod model (randomally choosen a compatible one)
        model = monopod.Monopod(world=task.world,
                                monopod_version=monopod_model)

        # Store the model name and model in the task
        task.model_name = model.name()
        task.model = model

        # RESET the monopod
        reset_orientation = np.random.choice(task.reset_positions)
        xpath = 'resets/' + reset_orientation
        task.current_reset_orientation = reset_orientation
        reset_conf = task.cfg.get_config(xpath)
        joint_angles = (0, 0)

        pos_reset = np.zeros(len(task.joint_names))
        vel_reset = np.zeros(len(task.joint_names))

        if task.task_mode is not 'simple':
            if not reset_conf['laying_down']:
                xpath = 'task_modes/' + task.task_mode + '/definition'
                robot_def = task.cfg.get_config(xpath)
                robot_def['planarizer_pitch_joint'] = reset_conf['planarizer_pitch_joint']
                joint_angles = leg_joint_angles(robot_def)
            else:
                joint_angles = (1.57,  0)

            # Get the model
            model = task.world.get_model(task.model_name)

            pos_reset[task.joint_names.index(
                'planarizer_pitch_joint')] = reset_conf['planarizer_pitch_joint']
        else:
            joint_angles = task.observation_space.sample()[[task.observation_index['hip_joint_pos'],task.observation_index['knee_joint_pos']]]

        pos_reset[task.joint_names.index('hip_joint')] = joint_angles[0]
        pos_reset[task.joint_names.index('knee_joint')] = joint_angles[1]
        ok_pos = model.to_gazebo().reset_joint_positions(
            pos_reset, task.joint_names)
        ok_vel = model.to_gazebo().reset_joint_velocities(
            vel_reset, task.joint_names)

        if not (ok_pos and ok_vel):
            raise RuntimeError("Failed to reset the monopod state")

        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def get_state_info(self, state: Observation, actions: Deque[Action]):
        return self.env.unwrapped.task.get_state_info(state, actions)
