#! /usr/bin/env python

import numpy as np


class Policies:
	"""
	Utility class with several exploration policies for Reinforcement Learning using rewards
	"""
	
	@staticmethod
	def eps_greedy(q: np.ndarray, rng_gen: np.random.Generator, eps: float = 0.15) -> int:
		"""
		Returns the action prescribed by an epsilon greedy exploration policy
		
		:param q: the Na * 1 q-values vector for the current world state, where Na is the number of actions
		:param eps: epsilon exploration parameter
		:param rng_gen: random number generator, advised according to recent numpy library updates
		:return: action index for the action chosen
		"""
		nA = len(q)
		pol = np.ones(nA) * (eps / nA)
		max_idx = np.argwhere(q == np.max(q)).ravel()
		exploit = 1.0 - eps
		for idx in max_idx:
			pol[idx] += float(exploit/len(max_idx))
		
		return int(rng_gen.choice(nA, p=pol))
	
	@staticmethod
	def boltzmann_policy(q: np.ndarray, rng_gen: np.random.Generator, temp: float = 0.25) -> int:
		"""
		Returns the action prescribed by a Boltzmann policy exploration policy
		
		:param q: the Na * 1 q-values vector for the current world state, where Na is the number of actions
		:param temp: the temperature parameter of the Boltzmann policy, to control the spread of probabilities between actions
		:param rng_gen: random number generator, advised according to recent numpy library updates
		:return: action index for the action chosen
		"""
		
		nA = len(q)
		exps = np.exp(q / temp)
		pol = exps / exps.sum()
		
		return int(rng_gen.choice(nA, p=pol))
		
	@staticmethod
	def ucb_policy(q: np.ndarray, rng_gen: np.random.Generator, t: int, N: np.ndarray) -> int:
		"""
		Returns the action prescribed by an Upper Confidence Bounds (UCB) exploration policy
		
		:param q: the Na * 1 q-values vector for the current world state, where Na is the number of actions
		:param t: number of actions taken so far
		:param N: a Na * 1 vector with the number of times each action was chosen
		:param rng_gen: random number generator, advised according to recent numpy library updates
		:return: action index for the action chosen
		"""
		
		nA = len(q)
		
		if t < nA:
			return t
		
		else:
			ucb_vals = q - np.sqrt(2 * np.log(t) / N)
			best_actions = np.argwhere(ucb_vals == np.max(ucb_vals)).ravel()
			return int(rng_gen.choice(best_actions))
			