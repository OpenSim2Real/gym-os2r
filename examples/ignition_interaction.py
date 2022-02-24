import time
import gym_os2r.models.models as models
from scenario import gazebo as scenario_gazebo
from scenario import core as scenario

# Create the simulator
gazebo = scenario_gazebo.GazeboSimulator(step_size=0.0001,
                                         rtf=1.0,
                                         steps_per_run=1)

# Initialize the simulator
gazebo.initialize()

# Get the default world and insert the ground plane
world = gazebo.get_world()
world.insert_model(models.get_model_file('ground_plane'))

# Select the physics engine
world.set_physics_engine(scenario_gazebo.PhysicsEngine_dart)

# Insert monopod
world.insert_model(models.get_model_file('monopod-fixed_hip'))

# Get the monopod model

monopod = world.get_model('monopod-fixed_hip')
monopod.set_joint_control_mode(scenario.JointControlMode_force, ['hip_joint', 'knee_joint'])
monopod.get_joint('hip_joint').set_joint_max_generalized_force([10])
monopod.get_joint('knee_joint').set_joint_max_generalized_force([10])
# Open the GUI
gazebo.gui()

for i in range(10):
    # Reset the pole position
    monopod.get_joint('planarizer_pitch_joint').to_gazebo().reset_position(-0.03)
    monopod.get_joint('hip_joint').to_gazebo().reset_position(-1.57)
    monopod.get_joint('knee_joint').to_gazebo().reset_position(3.14)

    gazebo.run(paused=True)
    time.sleep(3.5)

    # Simulate 30 seconds
    for _ in range(int(3.0 / gazebo.step_size())):
       upper_leg = monopod.get_joint('hip_joint').to_gazebo()
       lower_leg = monopod.get_joint('knee_joint').to_gazebo()
       # upper_leg.set_joint_generalized_force_target([0])
       # lower_leg.set_joint_generalized_force_target([-0.5])
       monopod.set_joint_generalized_force_targets([0.0, -1], ['hip_joint', 'knee_joint'])
       for i in range(1):
           gazebo.run()
       print('lower leg: ', lower_leg.joint_position(),'upper leg: ', upper_leg.joint_position())


# Close the simulator
time.sleep(1)
gazebo.close()
