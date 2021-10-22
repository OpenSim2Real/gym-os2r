import gym
import time
import functools
from gym_ignition.utils import logger

from gym_bb import randomizers
from gym_bb.common.make_envs import make_env_from_id
from gym_bb.rewards.rewards import BalancingV2


# Set verbosity
logger.set_level(gym.logger.ERROR)
# logger.set_level(gym.logger.DEBUG)

# Available tasks
env_id = "Monopod-balance-v1"

# Create a partial function passing the environment id
# kwargs = {'task_mode': 'fixed_hip_and_boom_yaw'}
# kwargs = {'task_mode': 'free_hip'}
# kwargs = {'reset_positions': ['stand', 'ground', 'lay', 'float']}
# kwargs = {'reset_positions': ['float']}
# kwargs = {'reward_class': BalancingV2}
kwargs = {}
make_env = functools.partial(make_env_from_id, env_id=env_id, **kwargs)
# env = randomizers.monopod_no_rand.MonopodEnvNoRandomizer(env=make_env)
env = randomizers.monopod_no_rand.MonopodEnvNoRandomizer(env=make_env)
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
        action = env.action_space.sample() * 0.1
        observation, reward, done, _ = env.step(action)
        # time.sleep(0.005)

env.close()
time.sleep(5)
