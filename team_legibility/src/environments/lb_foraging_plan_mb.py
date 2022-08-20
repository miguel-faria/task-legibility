#! /usr/bin/env python
import itertools

import numpy as np
import pandas as pd
import pickle
import os

from typing import List, NamedTuple, Tuple, Dict, Callable
from termcolor import colored
from scipy.sparse import csr_matrix
from collections import Counter
from utils import adjacent_locs
from enum import Enum
from pathlib import Path


OBSERVATION_RANGE = 9
UP_OBS = [1, 2, 3]
DOWN_OBS = [6, 7, 8]
LEFT_OBS = [1, 4, 6]
RIGHT_OBS = [3, 5, 8]
MIN_EXPLORATION = 0.1


class RelativePositions(Enum):
	NONE = 0
	ABOVE_LEFT = 1
	ABOVE = 2
	ABOVE_RIGHT = 3
	LEFT = 4
	RIGHT = 5
	BELOW_LEFT = 6
	BELOW = 7
	BELOW_RIGHT = 8


class FoodStates(Enum):
	NO_FOOD = 0
	CAN_PICK = 1
	PICKED = 2


class AgentRoles(Enum):
	NO_ROLE = 0
	LEADER = 1
	FOLLOWER = 2


def backup_model(file_path: Path, model: Dict, iteration: int) -> None:
	# Create dictionary with model data
	save_data = dict()
	save_data['q'] = model['q']
	save_data['p'] = model['P']
	save_data['c_est'] = model['c']
	save_data['run_it'] = iteration
	
	# Save data to file
	with open(file_path, 'wb') as pickle_file:
		pickle.dump(save_data, pickle_file)
		

def load_model(file_path: Path) -> Tuple[List, np.ndarray, np.ndarray, int]:
	# Load model from file
	with open(file_path, 'rb') as pickle_file:
		data = pickle.load(pickle_file)
	
	return data['p'], data['c_est'], data['q'], data['run_it']


class MBForagingPlan:

	def __init__(self, field_size: Tuple, max_n_food: int, max_food_level: int, n_agents: int, agent_level: List[int], agent_roles: List[int], min_food_level: int = 1):
		
		self._field_size = field_size
		self._max_n_food = max_n_food
		self._food_level = max_food_level
		self._n_agents = n_agents
		self._agent_levels = agent_level
		self._agent_roles = agent_roles
		self._min_level = min_food_level
		self._states = ()
		self._actions = ()
		self._transitions = {}
		self._rewards = {}
		self._rewards_est = {}
		self._optimal_q = {}
	
	@property
	def states(self) -> Tuple:
		return self._states
	
	@property
	def actions(self) -> Tuple:
		return self._actions
	
	@property
	def transitions(self) -> Dict:
		return self._transitions
	
	@property
	def rewards(self) -> Dict:
		return self._rewards
	
	@property
	def agent_roles(self) -> List[int]:
		return self._agent_roles
	
	@property
	def n_agents(self) -> int:
		return self._n_agents
	
	@property
	def agent_levels(self) -> List[int]:
		return self._agent_levels
	
	@property
	def food_level(self) -> int:
		return self._food_level
	
	@property
	def field_size(self) -> Tuple:
		return self._field_size
	
	def generate_states(self, agent_level: int) -> Tuple:
		
		locs = []
		rows, cols = self._field_size
		for i in range(rows):
			for j in range(cols):
				locs += [(i, j)]
	
		states = ()
		for loc in locs:
			skip_up = (loc[0] == 0)
			skip_down = (loc[0] == (rows - 1))
			skip_left = (loc[1] == 0)
			skip_right = (loc[1] == (cols - 1))
			for comb in itertools.product(range(OBSERVATION_RANGE), repeat=(self._n_agents - 1)):
				if ((skip_up and any([x in comb for x in UP_OBS])) or (skip_down and any([x in comb for x in DOWN_OBS])) or
						(skip_left and any([x in comb for x in LEFT_OBS])) or (skip_right and any([x in comb for x in RIGHT_OBS]))):
					continue
				go_next = False
				for j in range(len(comb)):
					for k in range(j+1, len(comb)):
						if j == k:
							go_next = True
				if not go_next:
					loc_str = ','.join([str(x) for x in loc]) + ',' + str(agent_level) + ',' + ','.join([str(x) for x in comb])
					states += (loc_str + ',0', loc_str + ',1', loc_str + ',2', )
			
		return states
	
	@staticmethod
	def get_state_tuple(state: str) -> Tuple[Tuple, int, Tuple, int]:
		
		state_split = state.split(',')
		loc = (int(state_split[0]), int(state_split[1]))
		level = int(state_split[2])
		rel_pos = tuple([int(x) for x in state_split[3:-1]])
		food_state = int(state_split[-1])
		
		return loc, level, rel_pos, food_state
	
	def can_load(self, fruit_loc: Tuple, fruit_level: int, agents: List[Tuple]) -> Tuple[bool, List]:
		
		adj_locs = adjacent_locs(fruit_loc, self._field_size)
		while fruit_loc in adj_locs:
			adj_locs.remove(fruit_loc)
		agent_levels = 0
		adj_agents = []
		
		for agent in agents:
			agent_loc = agent[0]
			if agent_loc in adj_locs:
				agent_levels += agent[1]
				adj_agents += [agent_loc]
		
		return fruit_level <= agent_levels, adj_agents
	
	def generate_actions(self) -> Tuple:
		raise NotImplementedError('Specialized model versions should implement the transitions')
	
	def generate_rewards(self, states: Tuple, actions: Tuple, food: Tuple) -> np.ndarray:
		raise NotImplementedError('Specialized model versions should implement the transitions')
	
	def sample_transitions(self, agents_states: List[int], a: int, agent_idx: int, food: Tuple) -> Tuple[float, int]:
		raise NotImplementedError('Specialized model versions should implement the transitions')
	
	def learn_dynamics(self, food: Tuple, n: int, gamma: float, action_func: Callable, rng_gen: np.random.Generator, explore_param: float, qinit: np.ndarray = None,
					   Pinit: np.ndarray = None, cinit: np.ndarray = None):
		raise NotImplementedError('Specialized model versions should implement the dynamics learning method')
	
	def generate_world_food(self, food: Tuple, action_func: Callable, rng_gen: np.random.Generator, explore_param: float, max_steps: int, gamma: float) -> None:
		raise NotImplementedError('Specialized model versions should implement the world generation method')
	
	
class MultiAgentMBForagingPlan(MBForagingPlan):
	
	def __init__(self, field_size: Tuple, max_n_food: int, max_food_level: int, n_agents: int, agent_level: List[int], agents_roles: List[int], min_food_level: int = 1):
		super().__init__(field_size, max_n_food, max_food_level, n_agents, agent_level, agents_roles, min_food_level)
	
	def generate_actions(self) -> Tuple:
		
		single_agent_actions = ['N', 'U', 'D', 'L', 'R', 'Lo']
		
		return tuple(single_agent_actions)
		
	def generate_rewards(self, states: Tuple, actions: Tuple, food: Tuple) -> np.ndarray:
		nX = len(states)
		nA = len(actions)
		roles = list(set(self._agent_roles))
		roles.sort()
		nR = len(roles)
		r = np.zeros((nR, nX, nA))
		
		food_loc = food[:-1]
		food_lvl = food[-1]
		food_adj_locs = adjacent_locs(food_loc, self._field_size)
		
		for state in states:
			agent_loc, agent_lvl, rel_pos, food_state = self.get_state_tuple(state)
			agent_adj = (agent_loc in food_adj_locs)
			if agent_adj:
				agent_rel_pos = np.array(agent_loc) - np.array(food_loc)
				for role in roles:
					if ((role == AgentRoles.LEADER and agent_rel_pos[1] == 0) or
							(role == AgentRoles.FOLLOWER and agent_rel_pos[0] == 0) or
							role == AgentRoles.NO_ROLE):
						if food_state == FoodStates.PICKED:
							r[roles.index(role), states.index(state), :] = 1.0
						elif food_state == FoodStates.CAN_PICK:
							r[roles.index(role), states.index(state), :] = 0.1
		
		return r
	
	def sample_transitions(self, agents_states: List[int], action: Tuple[str], agent_idx: int, food: Tuple) -> Tuple[float, int]:
		
		def get_relative_positions() -> List[int]:
		
			relative_locs = np.array(nxt_o_agents_loc) - np.array(nxt_agent_loc)
			relative_pos = []
			for loc in relative_locs:
				# Other agent is within 1 space of agent
				if sum(np.abs(loc)) < 3:
					# Other agent is above agent
					if loc[0] == -1:
						if loc[1] == -1:
							relative_pos += [RelativePositions.ABOVE_LEFT]  # Above left of agent
						elif loc[1] == 1:
							relative_pos += [RelativePositions.ABOVE_RIGHT]  # Above right of agent
						else:
							relative_pos += [RelativePositions.ABOVE]  # Directly above the agent
					# Other agent is below agent
					elif loc[0] == 1:
						if loc[1] == -1:
							relative_pos += [RelativePositions.BELOW_LEFT]  # Below left of agent
						elif loc[1] == 1:
							relative_pos += [RelativePositions.BELOW_RIGHT]  # Below right of agent
						else:
							relative_pos += [RelativePositions.BELOW]  # Directly below the agent
					# Other agent is to the sides of the agent
					else:
						if loc[1] == -1:
							relative_pos += [RelativePositions.LEFT]	 # Left of agent
						elif loc[1] == 1:
							relative_pos += [RelativePositions.RIGHT]  # Right of agent
			
			# There isn't any other agent around agent
			if len(relative_pos) < 1:
				relative_pos = [RelativePositions.NONE]
			
			return relative_pos
		
		def get_next_food_state() -> int:
		
			food_rel_loc = np.array(food_loc) - np.array(nxt_agent_loc)
			agents_rel_locs = np.array(nxt_o_agents_loc) - np.array(nxt_agent_loc)
			
			# Agent is adjacent to target food
			if sum(np.abs(food_rel_loc)) == 1:
				
				# Not enough agents are trying to pick food
				if sum(loading_agents) < food_lvl:
					if food_rel_loc[0] == -1:
						if RelativePositions.ABOVE_LEFT in nxt_rel_pos or RelativePositions.ABOVE_RIGHT in nxt_rel_pos or (-2, 0) in agents_rel_locs:
							return FoodStates.CAN_PICK
						else:
							return FoodStates.NO_FOOD
					elif food_rel_loc[0] == 1:
						if RelativePositions.BELOW_LEFT in nxt_rel_pos or RelativePositions.BELOW_RIGHT in nxt_rel_pos or (2, 0) in agents_rel_locs:
							return FoodStates.CAN_PICK
						else:
							return FoodStates.NO_FOOD
					elif food_rel_loc[1] == -1:
						if RelativePositions.ABOVE_LEFT in nxt_rel_pos or RelativePositions.BELOW_LEFT in nxt_rel_pos or (0, -2) in agents_rel_locs:
							return FoodStates.CAN_PICK
						else:
							return FoodStates.NO_FOOD
					else:
						if RelativePositions.ABOVE_RIGHT in nxt_rel_pos or RelativePositions.BELOW_RIGHT in nxt_rel_pos or (0, 2) in agents_rel_locs:
							return FoodStates.CAN_PICK
						else:
							return FoodStates.NO_FOOD
						
				# Enough agents are trying to pick food
				else:
					return FoodStates.PICKED
			
			# Agent is not adjacent to target food
			else:
				if food_state == FoodStates.PICKED:
					return FoodStates.NO_FOOD
				else:
					return food_state
		
		food_loc = food[:-1]
		food_lvl = food[-1]
		food_adj_locs = adjacent_locs(food_loc, self._field_size)
		max_rows, max_cols = self._field_size
		agent_state = self._states[agents_states[agent_idx]]
		agent_loc, agent_lvl, rel_pos, food_state = self.get_state_tuple(agent_state)
		n_agents = len(action)
		nxt_agent_loc = ()
		curr_o_agents_loc = []
		nxt_o_agents_loc = []
		loading_agents = []
		
		# Find next agent positions
		for act_idx in range(n_agents):
			agent_act = action[act_idx]
			if act_idx == agent_idx:
				if agent_act == 'U':
					nxt_agent_loc = (max(agent_loc[0] - 1, 0), agent_loc[1])
				elif agent_act == 'D':
					nxt_agent_loc = (min(agent_loc[0] + 1, max_rows - 1), agent_loc[1])
				elif agent_act == 'L':
					nxt_agent_loc = (agent_loc[0], max(agent_loc[1] - 1, 0))
				elif agent_act == 'R':
					nxt_agent_loc = (agent_loc[0], min(agent_loc[1] + 1, max_cols - 1))
				elif agent_act == 'Lo':
					nxt_agent_loc = agent_loc
					if agent_loc in food_adj_locs:
						loading_agents += [agent_lvl]
				else:
					nxt_agent_loc = agent_loc
			else:
				o_agent_loc, o_agent_lvl, o_rel_pos, o_food_state = self.get_state_tuple(self._states[agents_states[act_idx]])
				curr_o_agents_loc += [o_agent_loc]
				if agent_act == 'U':
					nxt_o_agents_loc += [(max(o_agent_loc[0] - 1, 0), o_agent_loc[1])]
				elif agent_act == 'D':
					nxt_o_agents_loc += [(min(o_agent_loc[0] + 1, max_rows - 1), o_agent_loc[1])]
				elif agent_act == 'L':
					nxt_o_agents_loc += [(o_agent_loc[0], max(o_agent_loc[1] - 1, 0))]
				elif agent_act == 'R':
					nxt_o_agents_loc += [(o_agent_loc[0], min(o_agent_loc[1] + 1, max_cols - 1))]
				elif agent_act == 'Lo':
					nxt_o_agents_loc += [o_agent_loc]
					if o_agent_loc in food_adj_locs:
						loading_agents += [agent_lvl]
				else:
					nxt_o_agents_loc += [o_agent_loc]
					
		# Verify possible colisions
		for i in range(len(nxt_o_agents_loc)):
			if nxt_o_agents_loc[i] == nxt_agent_loc:
				nxt_agent_loc = agent_loc
				nxt_o_agents_loc[i] = curr_o_agents_loc[i]
				
		# Check food state
		if food_state == 2:
			nxt_rel_pos = get_relative_positions()
			nxt_state_str = ','.join([str(x) for x in nxt_agent_loc]) + ',' + str(agent_lvl) + ',' + ','.join([str(x) for x in nxt_rel_pos]) + ',' + str(food_state)
		else:
			nxt_rel_pos = get_relative_positions()
			nxt_food_state = get_next_food_state()
			nxt_state_str = ','.join([str(x) for x in nxt_agent_loc]) + ',' + str(agent_lvl) + ',' + ','.join([str(x) for x in nxt_rel_pos]) + ',' + str(nxt_food_state)
			
		nxt_state = self._states.index(nxt_state_str)
		nxt_reward = self._rewards[''.join([str(x) for x in food[:-1]])][agents_states[agent_idx], self._actions.index(action[agent_idx])]
		return nxt_reward, nxt_state
	
	def learn_dynamics(self, food: Tuple, max_runs: int, gamma: float, action_func: Callable, rng_gen: np.random.Generator, explore_param: float, qinit: np.ndarray = None,
					   Pinit: Tuple = None, cinit: np.ndarray = None, finish: bool = False) -> Tuple[Tuple, np.ndarray, np.ndarray]:
		
		X = self._states
		A = self._actions
		diff_roles = list(set(self._agent_roles))
		diff_roles.sort()
		nX = len(X)
		nA = len(A)
		nR = len(diff_roles)
		
		models_dir = Path(__file__).parent.absolute().parent.absolute() / 'models'
		savepoint_file = models_dir / ('lb_foraging_mb_' + str(self.field_size[0]) + 'x' + str(self.field_size[1]) + '_a' + str(self.n_agents) + 'l' +
									   str(self.food_level) + '.model_bck')
		if finish:
			if savepoint_file.exists() and os.path.getsize(savepoint_file) > 0:
				P, c, q, start = load_model(savepoint_file)
			else:
				start = 0
				P = ()
				for i in range(nA):
					act_P = ()
					for j in range(nR):
						act_P += (np.eye(nX),)
					P += (act_P,)
				
				c = np.zeros((nR, nX, nA))
				q = np.zeros((nR, nX, nA))
			
		else:
			start = 0
			if Pinit is None:
				P = ()
				for i in range(nA):
					act_P = ()
					for j in range(nR):
						act_P += (np.eye(nX), )
					P += (act_P, )
			else:
				P = Pinit
			if cinit is None:
				c = np.zeros((nR, nX, nA))
			else:
				c = cinit
			if qinit is None:
				q = np.zeros((nR, nX, nA))
			else:
				q = qinit
		
		N = np.ones((nR, nX, nA))
		
		states = [np.random.choice(nX)]
		exploration = explore_param
		
		for t in range(start, max_runs):
			
			# Actions selection
			action = ()
			for agent_idx in self._n_agents:
				action += (self._actions[action_func(q[states[agent_idx], :], rng_gen, exploration)], )
				
			# Update model
			nxt_states = []
			for agent_idx in self._n_agents:
				
				x = states[agent_idx]
				a = self._actions.index(action[agent_idx])
				role_idx = diff_roles.index(self.agent_roles[agent_idx])
				ct, x1 = self.sample_transitions(states, action, agent_idx, food)
				nxt_states += [x1]
				
				N[role_idx, x, a] += 1
				P[a][role_idx][x, :] *= 1 - 1 / N[role_idx, x, a]
				P[a][role_idx][x, x1] += 1 / N[role_idx, x, a]
				
				c[role_idx, x, a] += (ct - c[role_idx, x, a]) / N[role_idx, x, a]
				
				q[role_idx, x, a] = c[role_idx, x, a] + gamma * P[a][role_idx][x, :].dot(q[role_idx].min(axis=1, keepdims=True))
			
			states = nxt_states
			exploration = max(MIN_EXPLORATION, exploration * 0.9999)
			
			if t == 0 or (t + 1) % 100 == 0:
				backup_model(savepoint_file, {'q': q, 'P': P, 'c': c}, t)
		
		return P, c, q
	
	def generate_world_food(self, food: Tuple, action_func: Callable, rng_gen: np.random.Generator, explore_param: float, max_steps: int, gamma: float, finish: bool = False) -> None:
		
		dict_idx = ''.join([str(x) for x in food[:-1]])
		
		if len(self._states) == 0:
			self._states = self.generate_states(self._agent_levels[0])
		
		if len(self._actions) == 0:
			self._actions = self.generate_actions()
		
		rewards = self.generate_rewards(self._states, self._actions, food)
		self._rewards[dict_idx] = rewards
		transitions, c_est, q = self.learn_dynamics(food, max_steps, gamma, action_func, rng_gen, explore_param, finish)
		
		self._transitions[dict_idx] = transitions
		self._rewards_est[dict_idx] = c_est
		self._optimal_q[dict_idx] = q

