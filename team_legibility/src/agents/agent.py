#! /usr/bin/env python

import numpy as np

from abc import ABC, abstractmethod
from typing import Callable, NamedTuple, List, Tuple
from learners.learner import Learner


class Timestep(NamedTuple):
	"""
	Timestep tuple for the lb-foraging scenario
	"""
	state: int
	action: int
	observation: Tuple
	feedback: float
	done: bool
	rng_gen: np.random.Generator
	policy_data: Tuple


class Agent(ABC):
	
	"""
	Base autonomous agent class
	"""
	
	def __init__(self, name: str, exploration_policy: Callable, learning_model: Learner):
		"""
		:param name: agent name
		:param exploration_policy: exploration policy to be used (look to Policies in policies.py for examples)
		:param learning_model: the type of learner the agent uses, must be of type learners.learner.Learner
		"""
		self._name = name
		self._policy = exploration_policy
		self._learner = learning_model
		self._state = None
		self._action = None
		
	def new_action(self, q: np.ndarray, rng_gen: np.random.Generator, *args) -> int:
		"""
		Returns the agent's action given the chosen exploration policy
		:param q: the Na * 1 q-values vector for the current world state, where Na is the number of actions
		:param rng_gen: random number generator, advised according to recent numpy library updates
		:return: action index
		"""
		return self._policy(q, rng_gen, *args)
		
	def train(self, timestep: NamedTuple) -> None:
		"""
		Perform a training step for the learning model
		:param timestep:
		:return: none
		"""
		self._learner.train_step(timestep)
		# print(self._learner.Q)
	
	@abstractmethod
	def eval(self, timestep: NamedTuple):
		"""
		#TODO
		:param timestep:
		:return:
		"""
		raise NotImplementedError()
	
	@abstractmethod
	def step(self, timestep: Timestep) -> int:
		"""
		Perform one agent step, updating q-table and choosing next action
		:param timestep:
		:return: action for agent to execute
		"""
		raise NotImplementedError()
	
	@property
	def name(self) -> str:
		return self._name
	
	@property
	def exploration_policy(self) -> Callable:
		return self._policy
	
	@property
	def learner_model(self) -> Learner:
		return self._learner
	
	@property
	def state(self) -> int:
		return self._state

	@property
	def action(self) -> int:
		return self._action
	
	@state.setter
	def state(self, state: int):
		self._state = state
	
	@action.setter
	def action(self, act: int):
		self._action = act
	
	@exploration_policy.setter
	def exploration_policy(self, policy: Callable):
		self._policy = policy
	
	@learner_model.setter
	def learner_model(self, learn_model: Learner):
		self._learner = learn_model
	