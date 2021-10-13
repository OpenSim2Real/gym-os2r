import time
import os.path as osp

from gym_bb.common.vec_env.vec_env import VecEnvWrapper
import numpy as np
from collections import deque

import matplotlib.pyplot as plt

from gym import Wrapper

plt.ion()


class MonitorPlot(Wrapper):
    EXT = "monitor.png"

    def __init__(self, env, plot_path=None, episodes_for_refresh=5,
                 save_every_num_epidoes=10):
        Wrapper.__init__(self, env)
        self.epret = None
        self.eplen = None
        self.epcount = 0
        self.episodes_for_refresh = episodes_for_refresh
        self.save_every_num_epidoes = save_every_num_epidoes
        self.tstart = time.time()

        # assert plot_path is not None
        self.saving = False if plot_path is None else True
        if not self.saving:
            self.plot_path = ''
        elif not plot_path.endswith(MonitorPlot.EXT):
            if osp.isdir(plot_path):
                self.plot_path = osp.join(plot_path, VecMonitorPlot.EXT)
            else:
                self.plot_path = plot_path + "." + VecMonitorPlot.EXT

        print('Plotting to the absolute path: ' + str(self.plot_path))
        self.dplt = dynplot(num_envs=1)
        self.episodes = []
        self.rewards = []

    def reset(self):
        obs = self.env.reset()
        self.epret = 0
        self.eplen = 0
        return obs

    def step(self,  action):
        obs, rew, done, info = self.env.step(action)
        newinfo = info.copy()
        self.epret += rew
        self.eplen += 1
        if done:
            ret = self.epret
            eplen = self.eplen
            epinfo = {'r': ret, 'l': eplen, 't': round(
                time.time() - self.tstart, 6)}
            newinfo['episode'] = epinfo
            self.epcount += 1
            self.epret = 0
            self.eplen = 0
            self.rewards.append(ret)
            if self.epcount % self.episodes_for_refresh == 0:
                self.dplt.plot(0, self.rewards)
                self.dplt.show()
            if self.epcount % self.save_every_num_epidoes == 0 and self.saving:
                self.dplt.plot(0, self.rewards)
                self.dplt.save_fig(self.plot_path)

        return obs, rew, done, newinfo

    def close(self):
        if self.saving:
            self.dplt.save_fig(self.plot_path)
        self.dplt.close_fig()
        return self.env.close()


class VecMonitorPlot(VecEnvWrapper):
    EXT = "monitor.png"
    f = None

    def __init__(self, venv, plot_path=None, keep_buf=0,
                 episodes_for_refresh=5, save_every_num_epidoes=10):
        VecEnvWrapper.__init__(self, venv)
        self.eprets = None
        self.eplens = None
        self.epcount = 0
        self.tstart = time.time()
        self.episodes_for_refresh = episodes_for_refresh
        self.save_every_num_epidoes = save_every_num_epidoes

        # assert plot_path is not None
        self.saving = False if plot_path is None else True
        if not self.saving:
            self.plot_path = ''
        elif not plot_path.endswith(VecMonitorPlot.EXT):
            if osp.isdir(plot_path):
                self.plot_path = osp.join(plot_path, VecMonitorPlot.EXT)
            else:
                self.plot_path = plot_path + "." + VecMonitorPlot.EXT

        print('Plotting to the absolute path: ' + str(self.plot_path))
        self.keep_buf = keep_buf
        if self.keep_buf:
            self.epret_buf = deque([], maxlen=keep_buf)
            self.eplen_buf = deque([], maxlen=keep_buf)
        self.dplt = dynplot(self.num_envs)
        self.episodes = [[] for _ in range(self.num_envs)]
        self.rewards = [[] for _ in range(self.num_envs)]

    def reset(self):
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, 'f')
        self.eplens = np.zeros(self.num_envs, 'i')
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.eprets += rews
        self.eplens += 1
        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                ret = self.eprets[i]
                eplen = self.eplens[i]
                epinfo = {'r': ret, 'l': eplen, 't': round(
                    time.time() - self.tstart, 6)}
                info['episode'] = epinfo
                if self.keep_buf:
                    self.epret_buf.append(ret)
                    self.eplen_buf.append(eplen)
                self.epcount += 1
                self.eprets[i] = 0
                self.eplens[i] = 0
                newinfos[i] = info
                # self.episodes[i].append(eplen)
                self.rewards[i].append(ret)
                # self.dplt.plot(self.episodes[i],self.rewards[i])
                self.dplt.plot(i, self.rewards[i])
        new_ep = any(dones)
        if new_ep and self.epcount % self.episodes_for_refresh == 0:
            self.dplt.show()
        if new_ep and self.epcount % self.save_every_num_epidoes == 0 \
                and self.saving:
            self.dplt.save_fig(self.plot_path)
        return obs, rews, dones, newinfos

    def close(self):
        if self.saving:
            self.dplt.save_fig(self.plot_path)
        self.dplt.close_fig()
        return self.venv.close()


class dynplot():
    def __init__(self, num_envs, refresh_rate=0.1):
        # Configure object
        self.refresh_rate = refresh_rate

    # Create figure and axis
        self.fig, self.ax = plt.subplots()

    # Set axis to auto-scale
        self.lines = []

        for _ in range(num_envs):
            line, = self.ax.plot([])
            self.lines.append(line)
        self.ax.set_autoscaley_on(True)
        self.ax.grid()

    def plot(self, i, y):
        self.lines[i].set_ydata(y)
        self.lines[i].set_xdata(np.arange(len(y)))

    def show(self, *args, **kwargs):

        # Rescale
        self.ax.relim()
        self.ax.autoscale_view()

        # Draw and flush
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show(*args, **kwargs)

    def save_fig(self, path):
        # Rescale
        self.ax.relim()
        self.ax.autoscale_view()

        # Draw and flush
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.fig.savefig(path)

    def close_fig(self):
        plt.close(self.fig)
