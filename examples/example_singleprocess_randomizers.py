import gym
import time
import functools
from gym_ignition.utils import logger
from BB_gym_Envs import randomizers

from gym_ignition.utils.typing import Action, Reward, Observation
# Set verbosity
logger.set_level(gym.logger.ERROR)
# logger.set_level(gym.logger.DEBUG)

# Available tasks
env_id = "Monopod-Gazebo-v1"


def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
    import gym
    import BB_gym_Envs
    return gym.make(env_id, **kwargs)


# Create a partial function passing the environment id
make_env = functools.partial(make_env_from_id, env_id=env_id)

# Wrap the environment with the randomizer.
# This is a simple example no randomization are applied.
# env = randomizers.monopod_no_rand.MonopodEnvNoRandomizations(env=make_env)

# # Wrap the environment with the randomizer.
# # This is a complex example that randomizes both the physics and the model.
# env = randomizers.monopod.MonopodEnvRandomizer(
#     env=make_env, seed=42, num_physics_rollouts=5)
# env = randomizers.monopod.MonopodEnvRandomizer(
#     env=make_env, num_physics_rollouts=5)

env = randomizers.monopod.MonopodEnvRandomizer(env=make_env)
# Enable the rendering
env.render('human')

# Initialize the seed
env.seed(42)

beg_time = time.time()
for epoch in range(1000):

    # Reset the environment
    observation = env.reset()

    # Initialize returned values
    done = False
    totalReward = 0

    while not done:

        # Execute a random action
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)

        # Render the environment.
        # It is not required to call this in the loop if physics is not randomized.
        # env.render('human')
        if done:
            print('rollout info: ', env.get_state_info(observation), ' Real Reward: ', reward)

        # Accumulate the reward
        totalReward += reward

        # Print the observation
        msg = ""
        for value in observation:
            msg += "\t%.6f" % value
        logger.debug(msg)

    print(f"Reward episode #{epoch}: {totalReward}")
print('time for 1000 episodes: ' + str(time.time()-beg_time))
env.close()
time.sleep(5)