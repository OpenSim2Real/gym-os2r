from gym_ignition.utils.typing import Reward, Observation


def get_reward(reward_type):
    if reward_type == 'Standing_v1':
        return standing_v1
    elif reward_type == 'Balancing_v1':
        return balancing_v1
    else:
        raise RuntimeError(
            'Reward type ' + reward_type + ' not supported in '
            'monopod environment.')


def supported_rewards():
    Supported_rewards = {
        'Standing_v1': ['free_hip', 'fixed_hip', 'fixed_hip_and_boom_yaw'],
        'Balancing_v1': ['free_hip', 'fixed_hip', 'fixed_hip_and_boom_yaw']
    }
    return Supported_rewards


"""
Balancing tasks. Start from standing and stay standing.
"""


def balancing_v1(obs: Observation) -> Reward:
    # Get vertical boom angle and velocity.
    bp = obs[2]
    return bp


"""
Standing tasks. Start from ground and stand up.
"""


def standing_v1(obs: Observation) -> Reward:
    # Get vertical boom angle and velocity.
    bp = obs[2]
    return bp


"""
Hopping tasks. Start either standing or from ground. favour circular movement.
"""
