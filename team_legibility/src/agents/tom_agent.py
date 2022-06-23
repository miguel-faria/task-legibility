#! /usr/bin/env python

import numpy as np
import termcolor
import time
import itertools

from tqdm import tqdm
from typing import List, Tuple


class ToMAgent(object):
	"""
		Agent that uses Theory of Mind (ToM) to identify the objectives an agent partner in the environment and act accordingly
	"""
	
	def __init__(self, q_library: List[np.ndarray], sign: int, sample_q_library: List[np.ndarray]):
		"""
		
		:param q_library:
		:param sign:
		"""
		self._q_library = q_library
		self._sign = sign
		self._interaction_likelihoods = []
		self._goal_prob = np.array([])
		self._task_list = []
		self._n_tasks = 0
		self._sample_q = sample_q_library
	
	@property
	def q_library(self) -> List:
		return self._q_library
	
	@property
	def q_sample(self) -> List:
		return self._sample_q
	
	@property
	def interaction_likelihoods(self) -> List:
		return self._interaction_likelihoods
	
	@property
	def goal_prob(self) -> np.ndarray:
		return self._goal_prob
	
	@property
	def task_list(self) -> List[int]:
		return self._task_list
	
	@q_library.setter
	def q_library(self, new_library: List[np.ndarray]):
		self._q_library = new_library
		
	@q_sample.setter
	def q_sample(self, new_sample_library: List[np.ndarray]):
		self._sample_q = new_sample_library
	
	def set_task_list(self, tasks: List[int]) -> None:
		"""
		
		:param tasks:
		:return:
		"""
		self._task_list = tasks
		self._n_tasks = len(tasks)
		self._goal_prob = np.ones(self._n_tasks) / self._n_tasks
	
	def task_inference(self) -> int:
		"""
		
		:return:
		"""
		if not self._task_list:
			print(termcolor.colored('List of possible tasks not defined!!', 'red'))
			return -1
		
		if len(self._interaction_likelihoods) > 0:
			likelihood = np.cumprod(np.array(self._interaction_likelihoods), axis=0)[-1]
		else:
			likelihood = np.zeros(self._n_tasks)
		likelihood_sum = likelihood.sum()
		if likelihood_sum == 0:
			p_max = np.ones(self._n_tasks) / self._n_tasks
		else:
			p_max = likelihood / likelihood.sum()
		high_likelihood = np.argwhere(p_max == np.amax(p_max)).ravel()
		return self._task_list[np.random.choice(high_likelihood)]
	
	def sample_probability(self, x: int, a: int, conf: float) -> np.ndarray:
		"""
		
		:param x:
		:param a:
		:param conf:
		:return:
		"""
		goals_likelihood = []
		
		for task_idx in self._task_list:
			q = self._sample_q[task_idx]
			goals_likelihood += [np.exp(self._sign * conf * (q[x, a] - np.max(q[x, :]))) / np.sum(np.exp(self._sign * conf * (q[x, :] - np.max(q[x, :]))))]
		
		goals_likelihood = np.array(goals_likelihood)
		return goals_likelihood
	
	def birl_inference(self, sample: Tuple, conf: float, rng_gen: np.random.Generator) -> Tuple[int, float]:
		"""
		
		:param sample:
		:param conf:
		:param rng_gen:
		:return:
		"""
		
		if not self._task_list:
			print(termcolor.colored('List of possible tasks not defined!!', 'red'))
			return -1, -1
		
		state, action = sample
		sample_prob = self.sample_probability(state, action, conf)
		likelihood = self._goal_prob * sample_prob
		self._goal_prob += sample_prob
		self._goal_prob = self._goal_prob / self._goal_prob.sum()
		self._interaction_likelihoods += [likelihood]
		
		r_likelihood = np.cumprod(np.array(self._interaction_likelihoods), axis=0)[-1]
		likelihood_sum = r_likelihood.sum()
		if likelihood_sum == 0:
			p_max = np.ones(self._n_tasks) / self._n_tasks
		else:
			p_max = r_likelihood / r_likelihood.sum()
		max_idx = np.argwhere(p_max == np.amax(p_max)).ravel()
		max_task_prob = rng_gen.choice(max_idx)
		task_conf = p_max[max_task_prob]
		task_idx = self._task_list[max_task_prob]
		
		return task_idx, task_conf
	
	def get_actions(self, task_idx: int, state: int, rng_gen: np.random.Generator) -> int:
		"""
		
		:param task_idx:
		:param state:
		:param rng_gen:
		:return:
		"""
		
		_, nA = self._q_library[task_idx].shape
		pol = np.isclose(self._q_library[task_idx], np.max(self._q_library[task_idx], axis=1, keepdims=True), rtol=1e-10, atol=1e-10).astype(int)
		pol = pol / pol.sum(axis=1, keepdims=True)
		
		return rng_gen.choice(nA, p=pol[state, :])
	
	def action(self, state: int, sample: Tuple, conf: float, rng_gen: np.random.Generator) -> int:
		"""
		
		:param state:
		:param sample:
		:param conf:
		:param rng_gen
		:return:
		"""
		
		predict_task, task_conf = self.birl_inference(sample, conf, rng_gen)
		action = self.get_actions(predict_task, state, rng_gen)
		
		return action
	
	def sub_acting(self, state: int, rng_gen: np.random.Generator, act_try: int, sample: Tuple, conf: float) -> int:
		"""
		Execute a sub-optimal action, useful in cases of a multi-agent deadlock occurrence
		:param state:
		:param rng_gen:
		:param act_try:
		:param sample:
		:param conf:
		:return:
		"""
		predict_task, task_conf = self.birl_inference(sample, conf, rng_gen)
		sorted_q = self._q_library[predict_task][state].copy()
		sorted_q.sort()
		
		if act_try > len(sorted_q):
			return rng_gen.choice(len(sorted_q))
		
		q_len = len(sorted_q)
		nth_best = sorted_q[max(-(act_try + 1), -q_len)]
		return rng_gen.choice(np.where(self._q_library[predict_task][state] == nth_best)[0])
	
	def reset_inference(self, tasks: List = None):
		"""
		
		:return:
		"""
		if tasks:
			self._task_list = tasks
			self._n_tasks = len(self._task_list)
		self._interaction_likelihoods = []
		self._goal_prob = np.ones(self._n_tasks) / self._n_tasks
		