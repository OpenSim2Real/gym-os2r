from gym_ignition.utils.typing import Reward, Observation


def standing_v1(obs: Observation) -> Reward:
    # Get vertical boom angle and velocity.
    bp = obs[2]
    return bp
