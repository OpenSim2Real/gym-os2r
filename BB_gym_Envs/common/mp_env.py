import gym
import numpy as np
import functools
from typing import Union

from BB_gym_Envs.common.vec_env import SubprocVecEnv
from BB_gym_Envs import randomizers

SupportedRandomizers = Union[randomizers.monopod_no_rand.MonopodEnvNoRandomizer, randomizers.monopod.MonopodEnvRandomizer]


def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
    import gym
    import BB_gym_Envs
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
