import time
import gym_bb.models.models as models
from scenario import gazebo as scenario_gazebo

# Create the simulator
gazebo = scenario_gazebo.GazeboSimulator(step_size=0.001,
                                         rtf=1.0,
                                         steps_per_run=1)

# Initialize the simulator
gazebo.initialize()

# Get the default world and insert the ground plane
world = gazebo.get_world()
world.insert_model(models.get_model_file("ground_plane"))

# Select the physics engine
world.set_physics_engine(scenario_gazebo.PhysicsEngine_dart)

# Insert monopod
world.insert_model(models.get_model_file("monopod"))

# Get the monopod model

monopod = world.get_model("monopod")
# Open the GUI
gazebo.gui()

# Reset the pole position
monopod.get_joint("planarizer_pitch_joint").to_gazebo().reset_position(0.2)

time.sleep(3.5)
gazebo.run(paused=True)

# Simulate 30 seconds
#for _ in range(int(30.0 / gazebo.step_size())):
#    upper_leg = monopod.get_joint("upper_leg_joint").to_gazebo()
#    lower_leg = monopod.get_joint("lower_leg_joint").to_gazebo()
#    upper_leg.set_joint_generalized_force_target([0.3])
#    lower_leg.set_joint_generalized_force_target([0.3])
#    gazebo.run()
#    print('lower leg: ', lower_leg.joint_position(),'upper leg: ', upper_leg.joint_position())


# Close the simulator
time.sleep(1000)
gazebo.close()
