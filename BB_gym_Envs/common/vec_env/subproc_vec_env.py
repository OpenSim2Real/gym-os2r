import multiprocessing as mp
from multiprocessing import Process, Pipe

import numpy as np
from .vec_env import VecEnv, CloudpickleWrapper


def worker(remote, parent_remote, env_fn):
	parent_remote.close()
	env = env_fn()
	while True:
		cmd, data = remote.recv()
		if cmd == 'step':
			ob, reward, done, info = env.step(data)
			if done:
			    ob = env.reset()
			remote.send((ob, reward, done, info))
		elif cmd == 'reset':
			remote.send(env.reset())
		elif cmd == 'render':
			remote.send(env.render())
		elif cmd == 'close':
			remote.close()
			break
		else:
			raise NotImplentedError

class SubprocVecEnv(VecEnv):
	def __init__(self, env_fns):
		self.waiting = False
		self.closed = False
		no_of_envs = len(env_fns)
		self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(no_of_envs)])
		self.ps = []
		if no_of_envs:
			self.action_space_single = env_fns[0]().action_space

		for wrk, rem, fn in zip(self.work_remotes, self.remotes, env_fns):
			proc = Process(target = worker, args = (wrk, rem, CloudpickleWrapper(fn)))
			self.ps.append(proc)

		for p in self.ps:
			p.daemon = True
			p.start()
		for remote in self.work_remotes:
			remote.close()

	def step_async(self, actions):
		if self.waiting:
			raise AlreadySteppingError
		self.waiting = True

		for remote, action in zip(self.remotes, actions):
			remote.send(('step', action))

	def step_wait(self):
		if not self.waiting:
			raise NotSteppingError

		results = [remote.recv() for remote in self.remotes]
		self.waiting = False
		obs, rews, dones, infos = zip(*results)
		return np.stack(obs), np.stack(rews), np.stack(dones), infos

	def step(self, actions):
		self.step_async(actions)
		return self.step_wait()

	def reset(self):
		for remote in self.remotes:
			remote.send(('reset', None))

		return np.stack([remote.recv() for remote in self.remotes])

	def close(self):
		if self.closed:
			return
		if self.waiting:
			for remote in self.remotes:
				remote.recv()
		for remote in self.remotes:
			remote.send(('close', None))
		for p in self.ps:
			p.join()
		self.closed = True
