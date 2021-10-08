import gym
import time
import functools
from gym_ignition.utils import logger
from gym_bb import randomizers
from gym_bb.common.make_envs import make_env_from_id

# Set verbosity
logger.set_level(gym.logger.ERROR)
# logger.set_level(gym.logger.DEBUG)

# Available tasks
env_id = "Monopod-balance-v1"

# Create a partial function passing the environment id
kwargs = {'task_mode': 'fixed_hip_and_boom_yaw'}
make_env = functools.partial(make_env_from_id, env_id=env_id, **kwargs)
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
            print('rollout info: ', env.get_state_info(
                observation), ' Real Reward: ', reward)

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
