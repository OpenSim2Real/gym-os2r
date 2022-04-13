import gym
import time
import functools

from gym_os2r import randomizers
from gym_os2r.common import make_env_from_id
from gym_os2r.models.config import SettingsConfig
# from gym_os2r.rewards import BalancingV3

from gym_ignition.utils import logger
# Set verbosity
logger.set_level(gym.logger.ERROR)
# logger.set_level(gym.logger.DEBUG)

reset_position = 'name_reset'

cfg = SettingsConfig()
xpath = 'resets/' + reset_position

new_pitch = 0.4
laying_down = True

cfg.set_config(laying_down, xpath + '/laying_down')
cfg.set_config(new_pitch, xpath + '/planarizer_pitch_joint')


env_id = "Monopod-stand-v1"
kwargs = {
'reset_positions': ['name_reset'],
'config': cfg
}

make_env = functools.partial(make_env_from_id, env_id=env_id, **kwargs)
env = randomizers.monopod_no_rand.MonopodEnvNoRandomizer(env=make_env)

# Enable the rendering
env.render('human')
# Initialize the seed
env.seed(42)

beg_time = time.time()
for epoch in range(1000):

    # Reset the environment
    observation = env.reset()
    # time.sleep(1)

    # Initialize returned values
    done = False
    totalReward = 0
    count = 0

    time.sleep(1)
    c = 0
    while not done:
        # Execute a random action
        # action = env.action_space.sample() * 0.1  # make the value smaller
        # action = env.action_space.sample()
        action = [-1, -1]
        # action = [0.0, 0.2]
        observation, reward, done, info = env.step(action)
        # done = True
        # time.sleep(0.05)
        c += 1
        done = c==400

    # while not done:
    #     count += 1
    #     # Execute a random action
    #     # action = env.action_space.sample() * 0.1  # make the value smaller
    #     action = env.action_space.sample()
    #     # action = [0.0, 0.2]
    #     observation, reward, done, _ = env.step(action)
    #
    #     if observation[2] < -0.1:
    #         print('Bad obs: ', observation)
    #         done = True
    #     # print('observations: ', observation)
    #     # print('obseration high: ', env.observation_space.high, 'obseration low: ', env.observation_space.low)
    #     # print(env.task.observation_index)
    #     if count == 1000:
    #         done = True

env.close()
time.sleep(5)
