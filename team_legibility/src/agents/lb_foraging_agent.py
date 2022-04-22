#! /usr/bin/env python

import numpy as np

from typing import Callable, NamedTuple, List, Tuple
from .agent import Agent, Timestep
from learners.learner import Learner
from learners.q_learner import QTimestep


class LBAgentState(NamedTuple):
	"""
	State tuple for an agent in the LB Foraging scenario
	"""
	field: str
	fruits: List
	agents: List


class LBForagingAgent(Agent):
	
	"""
	Agent class for the LB-Foraging scenario of Christianos et al. 2021 (https://github.com/semitable/lb-foraging)
	"""

	def __init__(self, exploration_policy: Callable, learning_model: Learner, n_fruits: int, n_agents: int, agent_idx: int, name='LB-Foraging-Agent'):
		super().__init__(name, exploration_policy, learning_model)
		self._agent_idx = agent_idx
		self._n_fruits = n_fruits
		self._n_agents = n_agents
		
	def _make_state_obs(self, observation: Tuple) -> LBAgentState:
	
		field = str(observation[1])
		for c in ["]", "[", " ", "\n"]:
			field = field.replace(c, "")
		agent_obs = observation[0]
		fruits = []
		agents = []
		for i in range(self._n_fruits):
			fruits += [(agent_obs[3 * i], agent_obs[3 * i + 1], agent_obs[3 * i + 2])]
		
		for i in range(self._n_agents):
			agents += [(agent_obs[3 * self._n_fruits + 3 * i], agent_obs[3 * self._n_fruits + 3 * i + 1], agent_obs[3 * self._n_fruits + 3 * i + 2])]
			
		return LBAgentState(field, fruits, agents)
	
	@staticmethod
	def _convert_state_obs(state_obs: LBAgentState) -> int:
		
		state_str = state_obs.field
		for agent in state_obs.agents:
			state_str += str(int(agent[0])) + str(int(agent[1])) + str(int(agent[2]))
			
		return int(state_str)
	
	def get_state_from_obs(self, observation: Tuple) -> int:
		state = self._make_state_obs(observation)
		return self._convert_state_obs(state)
	
	def eval(self, timestep: NamedTuple):
		pass
	
	def step(self, timestep: Timestep) -> int:
		pass
		
	@property
	def agent_idx(self) -> int:
		return self._agent_idx


class LBForagingQAgent(LBForagingAgent):
	
	"""
	Q-Learning agent for the LB-Foraging scenario of Christianos et al. 2021
	"""
	
	def __init__(self, exploration_policy: Callable, learning_model: Learner, n_fruits: int, n_agents: int, agent_idx: int):
		super().__init__(exploration_policy, learning_model, n_fruits, n_agents, agent_idx, 'LB-Foraging-Q-Agent')
		
	def step(self, timestep: Timestep) -> int:
		
		state = timestep.state
		action = timestep.action
		obs = timestep.observation
		feedback = timestep.feedback
		rng_gen = timestep.rng_gen
		policy_data = timestep.policy_data
		
		# Get next state from observation
		nxt_state = self._convert_state_obs(self._make_state_obs(obs))
		
		# Update Q-table
		q_timestep = QTimestep(state, action, feedback, nxt_state)
		self.train(q_timestep)
		
		# Choose next action to exectute
		nxt_action = self.new_action(self._learner.Q.loc[nxt_state, :].to_numpy(), rng_gen, *policy_data)
		
		self._state = nxt_state
		self._action = nxt_action
		
		return nxt_action
