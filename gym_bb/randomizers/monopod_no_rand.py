from typing import Union
import random
from gym_bb import tasks
from gym_bb.models import monopod
from gym_ignition.randomizers import gazebo_env_randomizer
from gym_ignition.randomizers.gazebo_env_randomizer import MakeEnvCallable
from gym_ignition.utils.typing import Observation
from gym_bb.utils.reset import leg_joint_angles

# Tasks that are supported by this randomizer. Used for type hinting.
SupportedTasks = Union[tasks.monopod.MonopodTask]


class MonopodEnvNoRandomizer(gazebo_env_randomizer.GazeboEnvRandomizer):
    """
    Dummy environment randomizer for monopod tasks.

    Check :py:class:`gym_bb.randomizers.monopod.MonopodRandomizersMixin`
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

        # Store the model name in the task
        task.model_name = model.name()

        # RESET the monopod
        reset_position = random.choice(task.reset_positions)
        xpath = 'resets/' + reset_position
        reset_conf = task.cfg.get_config(xpath)

        joint_angles = (0, 0)
        if not reset_conf['laying_down']:
            xpath = 'task_modes/' + task.task_mode + '/definition'
            robot_def = task.cfg.get_config(xpath)
            robot_def['boom_pitch_joint'] = reset_conf['boom_pitch_joint']
            joint_angles = leg_joint_angles(robot_def)
        else:
            joint_angles = (1.57,  0)

        # Get the model
        model = task.world.get_model(task.model_name)

        pos_reset = vel_reset = [0]*len(task.joint_names)
        pos_reset[task.joint_names.index(
            'boom_pitch_joint')] = reset_conf['boom_pitch_joint']
        pos_reset[task.joint_names.index('upper_leg_joint')] = joint_angles[0]
        pos_reset[task.joint_names.index('lower_leg_joint')] = joint_angles[1]

        ok_pos = model.to_gazebo().reset_joint_positions(
            pos_reset, task.joint_names)
        ok_vel = model.to_gazebo().reset_joint_velocities(
            vel_reset, task.joint_names)

        if not (ok_pos and ok_vel):
            raise RuntimeError("Failed to reset the monopod state")

        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def get_state_info(self, state: Observation):
        return self.env.unwrapped.task.get_state_info(state)
