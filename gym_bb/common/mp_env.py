import gym
import numpy as np
import functools
from typing import Union

from gym_bb.common.vec_env import SubprocVecEnv
from gym_bb import randomizers

SupportedRandomizers = Union[randomizers.monopod_no_rand.MonopodEnvNoRandomizer, randomizers.monopod.MonopodEnvRandomizer]


def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
    import gym
    import gym_bb
    return gym.make(env_id, **kwargs)

def make_mp_envs(env_id, nenvs, seed, randomizer: SupportedRandomizers, start_idx = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param nenvs: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param randomizer: (SupportedRandomizers) the env randomizer
    :param rank: (int) index of the subprocess
    """
    def make_env(rank):
        def fn():
            make_env = functools.partial(make_env_from_id, env_id=env_id)
            env = randomizer(env=make_env)
            env.seed(seed + rank)
            return env
        return fn
    return SubprocVecEnv([make_env(i + start_idx) for i in range(nenvs)])
