import gym
import numpy as np
import time
import functools
from gym_ignition.utils import logger
from gym_bb import randomizers

from gym_ignition.utils.typing import Action, Reward, Observation

env_id = "Monopod-v1"


def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
    import gym
    import gym_bb
    return gym.make(env_id, **kwargs)


make_env = functools.partial(make_env_from_id, env_id=env_id)

env = randomizers.monopod_no_rand.MonopodEnvNoRandomizer(
    env=make_env, reward_type='Balancing_v1')
env.seed(42)

# This initial reset existing causes the bad observation.
# Removing the reset here makes it good again
# observation = env.reset()

for epoch in range(1000):

    observation = env.reset()
    observation, reward, done, _ = env.step(np.array([0, 0]))
    observation = env.reset()

    done = False

    while not done:
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        print(observation)

env.close()
time.sleep(5)
