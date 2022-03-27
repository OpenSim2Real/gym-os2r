import abc
from typing import Union, Deque

import os
from lxml import etree
from operator import add
from functools import reduce
import numpy as np

from scenario import gazebo as scenario
from gym_ignition import randomizers, utils
from gym_ignition.randomizers import gazebo_env_randomizer
from gym_ignition.randomizers.gazebo_env_randomizer import MakeEnvCallable
from gym_ignition.randomizers.model.sdf import Method, Distribution, UniformParams
from gym_ignition.utils.typing import Observation, Action
from gym_ignition.utils import logger

from gym_os2r import tasks
from gym_os2r.models import monopod
from gym_os2r.utils.reset import leg_joint_angles

# Tasks that are supported by this randomizer. Used for type hinting.
SupportedTasks = Union[tasks.monopod.MonopodTask]


class MonopodRandomizersMixin(randomizers.abc.TaskRandomizer,
                              randomizers.abc.PhysicsRandomizer,
                              randomizers.abc.ModelDescriptionRandomizer,
                              abc.ABC):
    """
    Mixin that collects the implementation of task, model and physics
    randomizations for monopod environments.
    """

    def __init__(self, randomize_physics_after_rollouts: int = 0):

        # Initialize base classes
        randomizers.abc.TaskRandomizer.__init__(self)
        randomizers.abc.PhysicsRandomizer.__init__(
            self, randomize_after_rollouts_num=randomize_physics_after_rollouts)
        randomizers.abc.ModelDescriptionRandomizer.__init__(self)

        # SDF randomizer
        self._sdf_randomizer_monopod = None
        self._sdf_randomizer_ground = None

    # ===========================
    # PhysicsRandomizer interface
    # ===========================

    def get_engine(self):

        return scenario.PhysicsEngine_dart

    def randomize_physics(self, task: SupportedTasks, **kwargs) -> None:

        gravity_z = task.np_random.normal(loc=-9.8, scale=0.2)

        if not task.world.to_gazebo().set_gravity((0, 0, gravity_z)):
            raise RuntimeError("Failed to set the gravity")

    # ========================
    # TaskRandomizer interface
    # ========================

    def randomize_task(self, task: SupportedTasks, **kwargs) -> None:

        # Remove the model from the world
        self._clean_world(task=task)

        if "gazebo" not in kwargs:
            raise ValueError("gazebo kwarg not passed to the task randomizer")

        gazebo = kwargs["gazebo"]

        # Execute a paused run to process model removal
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

        # Generate a random model description
        random_model = self.randomize_model_description(task=task)
        random_ground = self.randomize_ground_description(task=task)

        # Insert a new model in the world
        self._populate_world(task=task, monopod_model=random_model,
                             ground_model=random_ground)

        reset_orientation = np.random.choice(task.reset_positions)
        xpath = 'resets/' + reset_orientation
        task.current_reset_orientation = reset_orientation
        reset_conf = task.cfg.get_config(xpath)
        # Randomization,
        reset_conf['planarizer_pitch_joint'] *= np.random.uniform(0.8, 1.2)
        if not reset_conf['laying_down']:
            xpath = 'task_modes/' + task.task_mode + '/definition'
            robot_def = task.cfg.get_config(xpath)
            robot_def['planarizer_pitch_joint'] = reset_conf['planarizer_pitch_joint']
            leg_angles = leg_joint_angles(robot_def)
            random_angles = np.abs(np.random.normal((0,0), 0.2))
            # choose randomally hip or knee to randomize first.
            leg_angles[0] =  leg_angles[0] + (leg_angles[0]>0 - leg_angles[0]<0)*max(random_angles)
            leg_angles[1] =  leg_angles[1] - (leg_angles[1]>0 - leg_angles[1]<0)*min(random_angles)
        else:
            leg_angles = np.array([1.57, 0]) - (np.random.uniform() < 0.5) * np.array([3.14,  0])
            random_angles = np.abs(np.random.normal((0,0), 0.2))
            # choose randomally hip or knee to randomize first.
            leg_angles[0] =  leg_angles[0] + (leg_angles[0]>0 - leg_angles[0]<0)*max(random_angles)
            leg_angles[1] =  leg_angles[1] - (leg_angles[1]>0 - leg_angles[1]<0)*min(random_angles)
        random_dir = 1 - (np.random.uniform() < 0.5) * 2
        leg_angles = [angle * random_dir for angle in leg_angles]
        yaw_position = np.random.uniform(-0.2, 0.2)

        # Get the model
        model = task.world.get_model(task.model_name)

        pos_reset = np.zeros(len(task.joint_names))
        vel_reset = np.zeros(len(task.joint_names))
        pos_reset[task.joint_names.index(
            'planarizer_pitch_joint')] = reset_conf['planarizer_pitch_joint']
        pos_reset[task.joint_names.index('planarizer_yaw_joint')] = yaw_position
        pos_reset[task.joint_names.index('hip_joint')] = leg_angles[0]
        pos_reset[task.joint_names.index('knee_joint')] = leg_angles[1]

        ok_pos = model.to_gazebo().reset_joint_positions(
            pos_reset, task.joint_names)
        ok_vel = model.to_gazebo().reset_joint_velocities(
            vel_reset, task.joint_names)

        if not (ok_pos and ok_vel):
            raise RuntimeError("Failed to reset the monopod state")

        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    # ====================================
    # ModelDescriptionRandomizer interface
    # ====================================

    def randomize_model_description(self, task: SupportedTasks, **kwargs) -> str:

        randomizer = self._get_sdf_randomizer_monopod(task=task)
        # print(randomizer.sample(pretty_print=True))
        sdf = utils.misc.string_to_file(randomizer.sample())
        return sdf

    def randomize_ground_description(self, task: SupportedTasks, **kwargs) -> str:

        randomizer = self._get_sdf_randomizer_ground(task=task)
        # print(randomizer.sample(pretty_print=True))
        sdf = utils.misc.string_to_file(randomizer.sample())
        return sdf

    # ===============
    # Private Methods
    # ===============

    def _get_sdf_randomizer_monopod(self, task: SupportedTasks) -> \
            randomizers.model.sdf.SDFRandomizer:

        if self._sdf_randomizer_monopod is not None:
            return self._sdf_randomizer_monopod

        xpath = 'task_modes/' + task.task_mode + '/model'
        monopod_model = task.cfg.get_config(xpath)
        # Get the model file
        urdf_model_file = monopod.get_model_file_from_name(monopod_model)

        # Convert the URDF to SDF
        sdf_model_string = scenario.urdffile_to_sdfstring(urdf_model_file)
        # Write the SDF string to a temp file
        sdf_model = utils.misc.string_to_file(sdf_model_string)

        # Create and initialize the randomizer
        sdf_randomizer = randomizers.model.sdf.SDFRandomizer(
            sdf_model=sdf_model)

        # Use the RNG of the task
        sdf_randomizer.rng = task.np_random

        randomization_config = {
            "*/link/inertial/mass": {
                'method': Method.Coefficient,
                'distribution': Distribution.Uniform,
                'params': UniformParams(low=0.8, high=1.2),
                'default_value': 0,
                'ignore_zeros': True,
                'force_positive': True,
            },
            "*/joint/axis/dynamics/friction": {
                'method': Method.Absolute,
                'distribution': Distribution.Uniform,
                'params': UniformParams(low=0.01, high=0.05),
                'default_value': 0,
                'ignore_zeros': False,  # We initialized the value as 0
                'force_positive': True,
            },
            "*/joint/axis/dynamics/damping": {
                'method': Method.Coefficient,
                'distribution': Distribution.Uniform,
                'params': UniformParams(low=0.8, high=1.2),
                'default_value': 0,
                'ignore_zeros': True,
                'force_positive': True,
            },
            "*/link/collision/surface/friction/ode/mu": {
                'method': Method.Coefficient,
                'distribution': Distribution.Uniform,
                'params': UniformParams(low=0.8, high=1.2),
                'default_value': 0.33,
                'ignore_zeros': False,
                'force_positive': True,
            }
        }

        def recursive_element_helper(path, randomizer):
            if path in ['', '*']:
                # Return if at base case (last element of xpath)
                return randomizer.find_xpath('*'), []
            path_split = os.path.split(path)
            # Depth first.
            elements, _ = recursive_element_helper(
                path_split[0], randomizer)
            """ Check if current path has elements in it.
             If current path has no elements then make them using one level
             down path. (loop invariant is the level below must exist)"""
            changed = []
            for element in elements:
                if element.find(path_split[1]) is None:
                    child = etree.SubElement(element, path_split[1])

                    if path_split[1] in ['collision']:
                        child.set("name", path_split[1])
                    if '/' not in path_split[1]:
                        changed.append(element)
                    logger.debug('Added the child ' + str(path_split[1])
                                 + ' to the sdf element ' + str(path_split[0]))
            changed = [element.findall(path_split[1]) for element in changed]
            changed = reduce(add, changed) if changed else changed
            new = reduce(add, [element.findall(path_split[1])
                               for element in elements])
            return new, changed

        # This class will make an element in the model xpath that didnt exist prior.
        def recursive_element_init(path, randomizer, default_value = 0):
            elements, changed_leaf = recursive_element_helper(path, randomizer)
            for element in changed_leaf:
                logger.debug('The leaf element added with the tag: '
                             + str(element.tag) + ' got set to the value 0')
                # element.text = str(0)
                element.text = str(default_value)

        for xpath, config in randomization_config.items():
            recursive_element_init(xpath, sdf_randomizer, config['default_value'])
            sdf_randomizer.new_randomization() \
                .at_xpath(xpath) \
                .method(config["method"]) \
                .sampled_from(config["distribution"], config['params']) \
                .force_positive(config["distribution"]) \
                .ignore_zeros(config["ignore_zeros"]) \
                .add()

        # Process the randomization
        sdf_randomizer.process_data()
        assert len(sdf_randomizer.get_active_randomizations()) > 0

        # Store and return the randomizer
        self._sdf_randomizer_monopod = sdf_randomizer
        return self._sdf_randomizer_monopod

    def _get_sdf_randomizer_ground(self, task: SupportedTasks) -> \
            randomizers.model.sdf.SDFRandomizer:

        if self._sdf_randomizer_ground is not None:
            return self._sdf_randomizer_ground
        # Get the model file
        urdf_model_file = monopod.get_model_file_from_name("ground_plane")

        # Convert the URDF to SDF
        sdf_model_string = scenario.urdffile_to_sdfstring(urdf_model_file)

        # Write the SDF string to a temp file
        sdf_model = utils.misc.string_to_file(sdf_model_string)

        # Create and initialize the randomizer
        sdf_randomizer = randomizers.model.sdf.SDFRandomizer(
            sdf_model=sdf_model)

        # Use the RNG of the task
        sdf_randomizer.rng = task.np_random

        randomization_config = {
            "*/link/collision/surface/friction/ode/mu": {
                'method': Method.Coefficient,
                'distribution': Distribution.Uniform,
                'params': UniformParams(low=0.8, high=1.2),
                'default_value': 0.33,
                'ignore_zeros': False,
                'force_positive': True,
            }
        }

        for xpath, config in randomization_config.items():
            sdf_randomizer.new_randomization() \
                .at_xpath(xpath) \
                .method(config["method"]) \
                .sampled_from(config["distribution"], config['params']) \
                .force_positive(config["distribution"]) \
                .ignore_zeros(config["ignore_zeros"]) \
                .add()

        # Process the randomization
        sdf_randomizer.process_data()
        assert len(sdf_randomizer.get_active_randomizations()) > 0

        # Store and return the randomizer
        self._sdf_randomizer_ground = sdf_randomizer
        return self._sdf_randomizer_ground

    @staticmethod
    def _clean_world(task: SupportedTasks) -> None:

        # Remove the model from the simulation
        if task.model_name is not None and task.model_name in task.world.model_names():

            if not task.world.to_gazebo().remove_model(task.model_name):
                raise RuntimeError(
                    "Failed to remove the monopod from the world")

        if "ground_plane" in task.world.model_names():

            if not task.world.to_gazebo().remove_model("ground_plane"):
                raise RuntimeError(
                    "Failed to remove the ground plane from the world")

    @staticmethod
    def _populate_world(task: SupportedTasks, monopod_model: str = None,
                        ground_model: str = None) -> None:
        #insert world
        if ground_model is None:
            ground_model = monopod.get_model_file_from_name("ground_plane")

        task.world.to_gazebo().insert_model(ground_model)
        # Insert a new monopod.
        # It will create a unique name if there are clashing.
        if monopod_model is None:
            xpath = 'task_modes/' + task.task_mode + '/model'
            monopod_model = task.cfg.get_config(xpath)

        model = monopod.Monopod(world=task.world, monopod_version=monopod_model,
                                model_file=monopod_model)

        # Store the model name in the task
        task.model_name = model.name()
        task.model = model


class MonopodEnvRandomizer(gazebo_env_randomizer.GazeboEnvRandomizer,
                           MonopodRandomizersMixin):
    """
    Concrete implementation of monopod environments randomization.
    """

    def __init__(self,
                 env: MakeEnvCallable,
                 num_physics_rollouts: int = 0,
                 **kwargs
                 ):

        # Initialize the mixin
        MonopodRandomizersMixin.__init__(
            self, randomize_physics_after_rollouts=num_physics_rollouts)

        # Initialize the environment randomizer
        gazebo_env_randomizer.GazeboEnvRandomizer.__init__(self, env=env,
                                                           physics_randomizer=self,
                                                           **kwargs)

    def get_state_info(self, state: Observation, actions: Deque[Action]):
        return self.env.unwrapped.task.get_state_info(state, actions)
