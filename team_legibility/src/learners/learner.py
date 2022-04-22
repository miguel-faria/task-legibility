#! /usr/bin/env python

import numpy as np

from abc import ABC, abstractmethod
from typing import NamedTuple, List
from pathlib import Path


class Learner(ABC):
	
	"""
	Base Learner Model class
	Represents the general concept of a learning model in Reinforcement Learning
	"""
	
	def __init__(self, name: str, feedback_type: str, actions: List):
		"""
		:param name: learner model name
		:param feedback_type: type of feedback signal used: costs or rewards
		:param actions: list of actions possible for the learner
		"""
		
		self._name = name
		self._feedback_type = feedback_type
		self._actions = actions
		self._actions_num = len(actions)
		self._Q = None
		
	@abstractmethod
	def train_step(self, timestep: NamedTuple) -> None:
		"""
		Executes one step of update for the learner
		:param timestep: tuple with the data required for the learner update
		:return: none
		"""
		raise NotImplementedError()
	
	@abstractmethod
	def save(self, filename: str) -> None:
		"""
		Save learned model to file
		:param filename: filename to save learned model
		:return: none
		"""
		raise NotImplementedError()
	
	@abstractmethod
	def load(self, filename: str) -> None:
		"""
		Load learned model from file
		:param filename: filename to load model from
		:return: none
		"""
		raise NotImplementedError()
	
	@property
	def name(self) -> str:
		return self._name
	
	@property
	def Q(self):
		return self._Q
	
	@property
	def actions(self) -> List:
		return self._actions
	
	@property
	def feedback_type(self) -> str:
		return self._feedback_type
	
	@feedback_type.setter
	def feedback_type(self, feedback: str):
		self._feedback_type = feedback
