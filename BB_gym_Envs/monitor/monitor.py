import time
from glob import glob
import csv
import os.path as osp
import json

from BB_gym_Envs.common.vec_env.vec_env import VecEnvWrapper
import numpy as np
import time
from collections import deque

import matplotlib.pyplot as plt

plt.ion()

class VecMonitorPlot(VecEnvWrapper):
	EXT = "monitor.png"
	f = None

	def __init__(self, venv, filename=None, keep_buf=0, episodes_for_refresh=1):
		VecEnvWrapper.__init__(self, venv)
		self.eprets = None
		self.eplens = None
		self.epcount = 0
		self.tstart = time.time()
		self.episodes_for_refresh = episodes_for_refresh

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
		# if epcount % 5:

		obs, rews, dones, infos = self.venv.step_wait()
		self.eprets += rews
		self.eplens += 1
		newinfos = list(infos[:])
		for i in range(len(dones)):
			if dones[i]:
				info = infos[i].copy()
				ret = self.eprets[i]
				eplen = self.eplens[i]
				epinfo = {'r': ret, 'l': eplen, 't': round(time.time() - self.tstart, 6)}
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

		if any(dones) and self.epcount % self.episodes_for_refresh == 0:
			self.dplt.show()
		return obs, rews, dones, newinfos

class VecMonitor(VecEnvWrapper):
	EXT = "monitor.csv"
	f = None

	def __init__(self, venv, filename=None, keep_buf=0, info_keywords=()):
		VecEnvWrapper.__init__(self, venv)
		self.eprets = None
		self.eplens = None
		self.epcount = 0
		self.tstart = time.time()
		if filename:
			self.results_writer = ResultsWriter(filename, header={'t_start': self.tstart},
				extra_keys=info_keywords)
		else:
			self.results_writer = None
		self.info_keywords = info_keywords
		self.keep_buf = keep_buf
		if self.keep_buf:
			self.epret_buf = deque([], maxlen=keep_buf)
			self.eplen_buf = deque([], maxlen=keep_buf)

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
				epinfo = {'r': ret, 'l': eplen, 't': round(time.time() - self.tstart, 6)}
				for k in self.info_keywords:
					epinfo[k] = info[k]
				info['episode'] = epinfo
				if self.keep_buf:
					self.epret_buf.append(ret)
					self.eplen_buf.append(eplen)
				self.epcount += 1
				self.eprets[i] = 0
				self.eplens[i] = 0
				if self.results_writer:
					self.results_writer.write_row(epinfo)
				newinfos[i] = info
		return obs, rews, dones, newinfos


class ResultsWriter(object):
	def __init__(self, filename, header='', extra_keys=()):
		self.extra_keys = extra_keys
		assert filename is not None
		if not filename.endswith(VecMonitor.EXT):
			if osp.isdir(filename):
				filename = osp.join(filename, VecMonitor.EXT)
			else:
				filename = filename + "." + VecMonitor.EXT
		self.f = open(filename, "wt")
		if isinstance(header, dict):
			header = '# {} \n'.format(json.dumps(header))
		self.f.write(header)
		self.logger = csv.DictWriter(self.f, fieldnames=('r', 'l', 't')+tuple(extra_keys))
		self.logger.writeheader()
		self.f.flush()

	def write_row(self, epinfo):
		if self.logger:
			self.logger.writerow(epinfo)
			self.f.flush()


def get_monitor_files(dir):
	return glob(osp.join(dir, "*" + VecMonitor.EXT))

def load_results(dir):
	import pandas
	monitor_files = (
		glob(osp.join(dir, "*monitor.json")) +
		glob(osp.join(dir, "*monitor.csv"))) # get both csv and (old) json files
	if not monitor_files:
		raise LoadMonitorResultsError("no monitor files of the form *%s found in %s" % (VecMonitor.EXT, dir))
	dfs = []
	headers = []
	for fname in monitor_files:
		with open(fname, 'rt') as fh:
			if fname.endswith('csv'):
				firstline = fh.readline()
				if not firstline:
					continue
				assert firstline[0] == '#'
				header = json.loads(firstline[1:])
				df = pandas.read_csv(fh, index_col=None)
				headers.append(header)
			elif fname.endswith('json'): # Deprecated json format
				episodes = []
				lines = fh.readlines()
				header = json.loads(lines[0])
				headers.append(header)
				for line in lines[1:]:
					episode = json.loads(line)
					episodes.append(episode)
				df = pandas.DataFrame(episodes)
			else:
				assert 0, 'unreachable'
			df['t'] += header['t_start']
		dfs.append(df)
	df = pandas.concat(dfs)
	df.sort_values('t', inplace=True)
	df.reset_index(inplace=True)
	df['t'] -= min(header['t_start'] for header in headers)
	df.headers = headers # HACK to preserve backwards compatibility
	return df

class dynplot():


	# TODO: Add and test more plotting functions
	supported_fcns = ['plot']

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


# dyn = dynplot(4)
# dyn.plot(0, [11,12,13])
# dyn.plot(1, [1,2,3])
# dyn.plot(2, [11])
# dyn.plot(3, [11,12,13, 5, 6, 7])
# dyn.show()
# time.sleep(10000)
# class dynplot():
#
#
# 	# TODO: Add and test more plotting functions
# 	supported_fcns = ['plot']
#
# 	def __init__(self, refresh_rate=0.1):
# 		# Configure object
# 		self.refresh_rate = refresh_rate
#
# 		# Create figure and axis
# 		self.fig, self.ax = plt.subplots()
#
# 		# Set axis to auto-scale
# 		self.ax.set_autoscaley_on(True)
#
# 		self._initialized = False
# 		self._crnt_line = 0
#
# 	def _update(self, fcn, *args, **kwargs):
# 		 # Create initial lines upon first calls
# 		if not self._initialized:
# 			lines = getattr(self.ax, fcn)(*args, **kwargs)
# 			# Verify if not consecutive call, adding multiple lines in
# 			# multiple steps (i.e. multiple calls of plot() before call to
# 			# show())
# 			if not hasattr(self, 'lines') or not self.lines:
# 				# create list if only one line existing
# 				if not isinstance(lines, list):
# 					lines = [lines]
#
# 				self.lines = lines
# 			else:
# 				for line in lines:
# 					self.lines.append(line)
#
# 		# Reuse existing lines upon following calls
# 		else:
# 			# Remove possible line styling indications from *args
# 			args = list(filter(lambda x: not isinstance(x, str), args))
#
# 			# Create set of lines to be updated
# 			if len(args) == 1:
# 				nbr_lines = 1
# 				single_line = True
# 			else:
# 				nbr_lines = len(args) // 2
# 				single_line = False
#
# 			# Only update parts of the lines
# 			if len(self.lines) > 1 and nbr_lines < len(self.lines):
# 				line_ids = list(range(self._crnt_line,
# 									  self._crnt_line + nbr_lines))
# 				line_ids = [i % len(self.lines) for i in line_ids]
#
# 				self._crnt_line = (line_ids[-1] + 1) % len(self.lines)
#
# 			# Update all lines
# 			else:
# 				line_ids = list(range(0, len(self.lines)))
# 				self._crnt_line = 0
# 			# Apply changes to set of lines to be updated
# 			for i, line_id in enumerate(line_ids):
# 				# Set line values
# 				if single_line:
# 					self.lines[line_id].set_ydata(args[i])
# 				else:
# 					self.lines[line_id].set_xdata(args[2*i])
# 					self.lines[line_id].set_ydata(args[2*i+1])
#
# 				# Set line attributes if existing
# 				for key, value in kwargs.items():
# 					getattr(self.lines[line_id], 'set_' + key)(value)
#
# 	def __getattr__(self, name):
# 		if name in self.supported_fcns:
#
# 			def wrapper(*args, **kwargs):
# 				return self._update(name, *args, **kwargs)
#
# 			return wrapper
#
# 	def show(self, permanent=False, *args, **kwargs):
# 		"""Displays figure
# 			Calls ``matplotlib.pyplot.pause()`` for continuous plotting or,
# 			if ``permanent`` is ``True`` forwards the call to
# 			 ``matplotlib.pyplot.show()``
# 			 :param permanent: Don't update or refresh plot
# 			 :type permanent: bool
# 		"""
# 		self._initialized = True
#
# 		# Rescale
# 		self.ax.relim()
# 		self.ax.autoscale_view()
#
# 		# Draw and flush
# 		self.fig.canvas.draw()
# 		self.fig.canvas.flush_events()
#
# 		if permanent:
# 			plt.show(*args, **kwargs)
# 		else:
# 			plt.pause(self.refresh_rate)


# import matplotlib.pyplot as plt
# plt.ion()
# from math import sin, pi
# import time
# dplt = dynplot()
#
#
# xdata = [[] for _ in range(10)]
# ydata = [[] for _ in range(10)]
# for x in np.arange(0,10,0.5):
# 	for i in range(10):
# 		xdata[i].append(x)
# 		ydata[i].append(i+np.exp(-x**2)+10*np.exp(-(x-7)**2))
# 		dplt.plot(xdata[i], ydata[i])
# 	dplt.show()
# 	time.sleep(1)
