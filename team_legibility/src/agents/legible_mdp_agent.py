#! /usr/bin/env python

from __future__ import annotations
import numpy as np

from typing import List, Tuple, Dict
from agents.mdp_agent import MDPAgent
from scipy.sparse import csr_matrix


class LegibleMDPAgent(MDPAgent):
	
	def __init__(self, x: Tuple, a: Tuple, p: List[List[csr_matrix]], gamma: float, verbose: bool, objectives: List[str], beta: float,
				 sign: int, q_mdps: List[np.ndarray], v_mdps: np.ndarray = None, dists: np.ndarray = None):
		self._objectives = objectives
		self._tasks_q = q_mdps
		self._v_mdps = v_mdps
		self._beta = beta
		nX = len(x)
		nA = len(a)
		nT = len(objectives)
		c = np.zeros((nT, nX, nA))
		
		super().__init__(x, a, p, c, gamma, 'rewards', verbose)
		for t in range(nT):
			for i in range(nX):
				for j in range(nA):
					c[t, i, j] = self.optimal_legible_cost(i, j, t, sign, dists)
		
		self._mdp = (x, a, p, c, gamma)
		
	def update_cost_function(self, sign: int, dists: np.ndarray):
		mdp = self.mdp
		x = mdp[0]
		a = mdp[1]
		p = mdp[2]
		gamma = mdp[4]
		nX = len(x)
		nA = len(a)
		nT = len(self._objectives)
		c = np.zeros((nT, nX, nA))
		
		for t in range(nT):
			for i in range(nX):
				for j in range(nA):
					c[t, i, j] = self.optimal_legible_cost(i, j, t, sign, dists)
		
		self._mdp = (x, a, p, c, gamma)
	
	def optimal_legible_cost(self, x: int, a: int, t: int, sign: int, dist: np.ndarray) -> float:
		task_cost = np.exp(sign * self._beta * self._tasks_q[t][x, a])
		tasks_sum = task_cost
		for task in range(len(self._objectives)):
			if task != t:
				tasks_sum += np.exp(sign * self._beta * self._tasks_q[task][x, a])
		
		return task_cost / tasks_sum
	
	def legible_cost(self, x: int, a: int, t: int, sign: int, dist: np.ndarray) -> float:
		task_cost = np.exp(sign * self._beta * self._tasks_q[t][x, a]) / (1 / dist[t, x])
		tasks_sum = task_cost
		for task in range(len(self._objectives)):
			if task != t:
				tasks_sum += np.exp(sign * self._beta * self._tasks_q[task][x, a]) / (1 / dist[task, x])
		
		return task_cost / tasks_sum
	
	def pol_legibility(self, tasks_q_pi: np.ndarray, task: int, eta: float) -> float:
		nX, nA = tasks_q_pi[task].shape
		
		task_prob = (np.exp(eta * (tasks_q_pi[task] - self._v_mdps[task][:, None])) / (nX * nA)).sum()
		task_prob_sum = []
		
		for objective in self._objectives:
			task_prob_sum += [(np.exp(eta * (tasks_q_pi[objective] - self._v_mdps[objective][:, None])) / (nX * nA)).sum()]
		
		task_prob_sum = np.array(task_prob_sum)
		return task_prob / task_prob_sum.sum()
	
	
