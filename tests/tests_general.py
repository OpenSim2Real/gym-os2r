from gym import envs
import functools
import numpy as np
from gym_bb import randomizers
from gym_bb.common.make_envs import make_mp_envs, make_env_from_id

from gym.spaces import Box
from gym.utils.env_checker import check_env
from gym_bb.rewards.reward_definition import BalancingV1, StandingV1


def check_registered_envs():

    all_envs = envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs if "Monopod" in env_spec.id]
    for env_id in env_ids:
        make_env = functools.partial(make_env_from_id, env_id=env_id)
        env = randomizers.monopod.MonopodEnvRandomizer(
            env=make_env, reward_class=BalancingV1)
        env.seed(42)
        # Test if env adheres to Gym API
        check_env(env, warn=True, skip_render_check=True)

        # Check that dtype is explicitly declared for gym.Box spaces

        ob_space = env.observation_space
        act_space = env.action_space
        ob = env.reset()
        assert ob_space.contains(
            ob), "Reset observation: {!r} not in space".format(ob)
        if isinstance(ob_space, Box):
            # Only checking dtypes for Box spaces to avoid iterating through tuple entries
            assert (
                ob.dtype == ob_space.dtype
            ), "Reset observation dtype: {}, expected: {}".format(ob.dtype, ob_space.dtype)

        a = act_space.sample()
        observation, reward, done, _info = env.step(a)
        assert ob_space.contains(observation), "Step observation: {!r} not in space".format(
            observation
        )
        assert np.isscalar(
            reward), "{} is not a scalar for {}".format(reward, env)
        assert isinstance(
            done, bool), "Expected {} to be a boolean".format(done)
        if isinstance(ob_space, Box):
            assert (
                observation.dtype == ob_space.dtype
            ), "Step observation dtype: {}, expected: {}".format(ob.dtype, ob_space.dtype)

        for mode in env.metadata.get("render.modes", []):
            env.render(mode=mode)

        # Make sure we can render the environment after close.
        for mode in env.metadata.get("render.modes", []):
            env.render(mode=mode)

        env.close()


def test_random_rollout():
    all_envs = envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs if "Monopod" in env_spec.id]
    for env_id in env_ids:
        make_env = functools.partial(make_env_from_id, env_id=env_id)
        env = randomizers.monopod.MonopodEnvRandomizer(
            env=make_env, reward_class=BalancingV1)
        env.seed(42)
        def agent(ob): return env.action_space.sample()
        ob = env.reset()
        for _ in range(10):
            assert env.observation_space.contains(ob)
            a = agent(ob)
            assert env.action_space.contains(a)
            (ob, _reward, done, _info) = env.step(a)
            if done:
                break
        env.close()


def single_process():
    env_id = "Monopod-v1"
    make_env = functools.partial(make_env_from_id, env_id=env_id)
    env = randomizers.monopod_no_rand.MonopodEnvNoRandomizer(
        env=make_env, reward_class=BalancingV1)
    env.seed(42)
    observation = env.reset()

    assert len(observation) == 10, "base monopod should have 10 observations"

    assert env.get_state_info(observation)[
                              1] == False, "Should not need reset after getting reset."

    action = env.action_space.sample()
    observation_after_step, reward, done, _ = env.step(action)

    assert env.get_state_info(observation_after_step)[
                              0] == reward, "should have same reward from step and get state info."
    assert all(observation_after_step
               != observation), "should have different observation after step."
    env.close()


def single_process_fixed_hip():
    env_id = "Monopod-fh-v1"
    make_env = functools.partial(make_env_from_id, env_id=env_id)
    env = randomizers.monopod_no_rand.MonopodEnvNoRandomizer(
        env=make_env, reward_class=BalancingV1)
    env.seed(42)
    observation = env.reset()

    assert len(observation) == 8, "Fixed hip monopod should have 8 observations"

    assert env.get_state_info(observation)[
                              1] == False, "Should not need reset after getting reset."

    action = env.action_space.sample()
    observation_after_step, reward, done, _ = env.step(action)
    assert env.get_state_info(observation_after_step)[
                              0] == reward, "should have same reward from step and get state info."
    assert all(observation_after_step
               != observation), "should have different observation after step."
    env.close()


def test_monopod_model():
    env_id = "Monopod-v1"
    make_env = functools.partial(make_env_from_id, env_id=env_id)
    env = randomizers.monopod.MonopodEnvRandomizer(
        env=make_env, reward_class=BalancingV1)
    env.seed(42)
    observation = env.reset()
    assert len(observation) == 10, "base monopod should have 10 observations"
    env.close()

    env_id = "Monopod-fh-v1"
    make_env = functools.partial(make_env_from_id, env_id=env_id)
    env = randomizers.monopod.MonopodEnvRandomizer(
        env=make_env, reward_class=BalancingV1)
    observation = env.reset()
    assert len(observation) == 8, "fixed hip monopod should have 8 observations"
    env.close()

    env_id = "Monopod-fh-fby-v1"
    make_env = functools.partial(make_env_from_id, env_id=env_id)
    env = randomizers.monopod.MonopodEnvRandomizer(
        env=make_env, reward_class=BalancingV1)
    observation = env.reset()
    assert len(
        observation) == 6, "fixed hip and fixed boom yaw monopod should have 6 observations"
    env.close()

# def multi_process_fixed_hip():
    # env_id = "Monopod-v1"
    # make_env = functools.partial(make_env_from_id, env_id=env_id)
    # env = randomizers.monopod_no_rand.MonopodEnvNoRandomizer(env=make_env, reward_class=BalancingV1)
    # env.seed(42)
    # observation = env.reset()
    #
    # assert len(observation) == 8, "Fixed hip monopod should have 8 observations"
    #
    # assert env.get_state_info(observation)[1] == False, "Should not need reset after getting reset."
    #
    # action = env.action_space.sample()
    # observation_after_step, reward, done, _ = env.step(action)
    #
    # assert env.get_state_info(observation_after_step)[0] == reward, "should have same reward from step and get state info."
    # assert observation_after_step != observation, "should have different observation after step."


if __name__ == "__main__":
    check_registered_envs()
    test_random_rollout()
    test_monopod_model()
    single_process()
    single_process_fixed_hip()
    print("Everything passed")
