#! /usr/bin/env python

import numpy as np
import pandas as pd
import pickle

from .learner import Learner
from typing import List, NamedTuple
from pathlib import Path


class QTimestep(NamedTuple):
	"""
	Named tuple for a timestep of Q-learning:
	state - current world state
	action - action agent executed
	feedback - feedback signal from performing action
	next_state - state the world evolved towards
	"""
	state: int
	action: int
	feedback: float
	next_state: int


class SARSATimestep(NamedTuple):
	"""
	Named tuple for a timestep of SARSA-learning:
	state - current world state
	action - action agent executed
	feedback - feedback signal from performing action
	next_state - state the world evolved towards
	next_action - action for agent to execute in next_state
	"""
	state: int
	action: int
	feedback: float
	next_state: int
	next_action: int


class QLearner(Learner):
	
	"""
	Class for a Q-Learning learner model
	"""
	
	def __init__(self, actions: List, learn_rate: float, discount_factor: float,
				 feedback_type: str, initial_q: pd.DataFrame = pd.DataFrame(), name='Q_Learner'):
		"""
		:param actions: list of actions possible for the learner
		:param learn_rate: float measuring how new updates influence learned q-values
		:param discount_factor: float measuring the impact of future feedback signals
		:param feedback_type: type of feedback signal used: costs or rewards
		:param initial_q: pandas dataframe with the initial q-values to consider (default is an empty dataframe)
		"""
		super().__init__(name, feedback_type, actions)
		self._learning_rate = learn_rate
		self._gamma = discount_factor
		if initial_q.empty:
			self._Q = pd.DataFrame(columns=actions, dtype=np.float64)
		else:
			self._Q = initial_q
	
	def _create_state_entry(self, state: int) -> None:
		"""
		Creates state entry in Q table if entry does not exist
		:param state: state to verify (and create) entry
		:return: none
		"""
		q = self._Q.to_numpy()
		if state not in self._Q.index:
			series = np.array([0.0] * self._actions_num)
			indexes = list(self._Q.index) + [state]
			q = np.vstack((q, series))
			self._Q = pd.DataFrame(q, columns=self._actions, index=indexes)
	
	def train_step(self, timestep: QTimestep) -> None:
		"""
		Performs one Q-Learning train step and updates Q-table accordingly
		:param timestep: namedtuple with (state, action, feedback, next_state) for current timestep update
		:return: none
		"""
		x = timestep.state
		nxt_x = timestep.next_state
		a = timestep.action
		r = timestep.feedback
		
		# guarantees that there is an entry in the Q table for both states
		self._create_state_entry(x)
		self._create_state_entry(nxt_x)
		
		self._Q.at[x, a] = self._Q.at[x, a] + self._learning_rate * (r + self._gamma * self._Q.loc[nxt_x, :].max() - self._Q.at[x, a])
	
	def save(self, filename: str) -> None:
	
		save_path = Path(__file__).parent.absolute().parent.absolute().parent.absolute() / 'models'
		if not save_path.is_dir():
			save_path.mkdir(parents=True, exist_ok=False)
		save_file = save_path / (filename + '.pkl')
		model = {'learn_rate': self._learning_rate, 'discount_factor': self._gamma, 'Q': self._Q}
		pickle.dump(model, open(save_file, 'wb'))
	
	def load(self, filename: str) -> None:
		
		try:
			load_file = Path(__file__).parent.absolute().parent.absolute().parent.absolute() / 'models' / (filename + '.pkl')
			model = pickle.load(open(load_file, 'rb'))
			self._learning_rate = model['learn_rate']
			self._gamma = model['discount_factor']
			self._Q = model['Q']
		except IOError:
			print('Could not access file')
	
	@property
	def Q(self) -> pd.DataFrame:
		return self._Q
	
	@property
	def learning_rate(self) -> float:
		return self._learning_rate
	
	@property
	def discount_factor(self) -> float:
		return self._gamma
	
	@learning_rate.setter
	def learning_rate(self, learn_rate):
		self._learning_rate = learn_rate
		
	@discount_factor.setter
	def discount_factor(self, gamma):
		self._gamma = gamma
	
	
class SARSALearner(QLearner):
	
	def __init__(self, actions: List, learn_rate: float, discount_factor: float,
				 feedback_type: str, initial_q: pd.DataFrame = pd.DataFrame()):
		super().__init__(actions, learn_rate, discount_factor, feedback_type, initial_q, 'SARSA')
		
	def train_step(self, timestep: SARSATimestep) -> None:
		x = timestep.state
		nxt_x = timestep.next_state
		a = timestep.action
		nxt_a = timestep.next_action
		r = timestep.feedback
		
		# guarantees that there is an entry in the Q table for both states
		self._create_state_entry(x)
		self._create_state_entry(nxt_x)
		
		self._Q.at[x, a] = self._Q.at[x, a] + self._learning_rate * (r + self._gamma * self._Q.at[nxt_x, nxt_a] - self._Q.at[x, a])
