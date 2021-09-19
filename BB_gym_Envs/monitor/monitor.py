import numpy as np
class TrainingMonitor(object):
    """docstring for TrainingMonitor."""

    def __init__(self, number_env, total_steps):
        super(TrainingMonitor, self).__init__()
        self.number_env = number_env
        self.training_data = np.empty((number_env, total_steps))
        self.current_episode_reward = np.zeros(number_env)
        self.last_episode_reward = np.zeros(number_env)
    def insert_data(self, id, data):
        ob, reward, done, info = data
        self.current_episode_reward[id] += reward
        print(self.current_episode_reward)
        if done:
            self.last_episode_reward[id] = self.current_episode_reward[id]
            self.current_episode_reward[id] = 0

    def get_current_episode_reward(self):
        return self.current_episode_reward

    def get_last_episode_reward(self):
        print(self.last_episode_reward)
        return self.last_episode_reward
