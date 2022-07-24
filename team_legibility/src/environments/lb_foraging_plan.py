#! /usr/bin/env python
import itertools

import numpy as np
import pandas as pd
import pickle

from typing import List, NamedTuple, Tuple, Dict
from termcolor import colored
from scipy.sparse import csr_matrix


class ForagingPlan:

	def __init__(self, field_size: Tuple, max_food: int, max_food_level: int, n_agents: int, agent_level: List[int], min_food_level: int = 1):
		
		self._field_size = field_size
		self._max_food = max_food
		self._max_level = max_food_level
		self._n_agents = n_agents
		self._agent_levels = agent_level
		self._min_level = min_food_level
		self._states = ()
		self._actions = ()
		self._transitions = {}
		self._rewards = {}
	
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
	
	def generate_states(self) -> Tuple:
		
		locs = []
		rows, cols = self._field_size
		for i in range(rows):
			for j in range(cols):
				locs += [(i, j)]
	
		states = ()
		for comb in itertools.product(locs, repeat=self._n_agents):
			add = True
			for i in range(self._n_agents):
				for j in range(i + 1, self._n_agents):
					if comb[i] == comb[j]:
						add = False
			if add:
				state_tmp = ''
				for i in range(self._n_agents):
					state_tmp += ''.join([str(x) for x in comb[i]]) + str(self._agent_levels[i])
				states += (state_tmp + '0', state_tmp + '1', )
			
		return states
	
	@staticmethod
	def get_state_tuple(state: str) -> Tuple[List[Tuple], List, int]:
		
		locs = []
		levels = []
		state_parse = list(state)
		
		# Iterate over state string separated, grouping agents positions and levels
		it = iter(state_parse[:-1])
		for x in it:
			locs += [(int(x), int(next(it)))]
			levels += [int(next(it))]
			
		return locs, levels, int(state_parse[-1])
	
	@staticmethod
	def adjacent_locs(state: Tuple, field_size: Tuple) -> List[Tuple]:
		
		state_row, state_col = state
		field_rows, field_cols = field_size
		return list({(min(state_row + 1, field_rows - 1), state_col), (max(state_row - 1, 0), state_col),
					 (state_row, max(state_col - 1, 0)), (state_row, min(state_col + 1, field_cols - 1))})
	
	def can_load(self, fruit_loc: Tuple, fruit_level: int, agents: List[Tuple]) -> Tuple[bool, List]:
		
		adj_locs = self.adjacent_locs(fruit_loc, self._field_size)
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
		return tuple(['N', 'U', 'D', 'L', 'R', 'Lo'])
	
	def generate_transitions(self, states: Tuple, actions: Tuple, food: Tuple) -> List[csr_matrix]:
		
		def verify_collisions(agents_states: Dict) -> List:
			
			nxt_state_list = []
			for loc in agents_states.keys():
				if len(agents_states[loc]) > 1:
					for agent in agents_states[loc]:
						if len(nxt_state_list) < agent[-1]:
							nxt_state_list += [list(agent[0:-1])]
						else:
							nxt_state_list = nxt_state_list[:agent[-1]] + [list(agent[0:-1])] + nxt_state_list[agent[-1]:]
				else:
					nxt_state_list += [[int(x) for x in loc] + [agents_states[loc][0][2]]]
			
			return nxt_state_list
	
		def compose_state(state_lst: List, fruit_got: int) -> str:
			
			comp_state = ''
			
			for s in state_lst:
				comp_state += ''.join([str(x) for x in s])
				
			comp_state += str(fruit_got)
			return comp_state
		
		nX = len(states)
		P = []
		n_rows, n_cols = self._field_size
		
		food_loc = food[:-1]
		food_lvl = food[-1]
		
		for act in actions:
			
			p = np.zeros((nX, nX))
			for state in states:
				
				state_idx = states.index(state)
				agent_states, agent_levels, fruit_pick = self.get_state_tuple(state)
				nxt_agent_states = []
				for agent_state in agent_states[1:]:
					nxt_agent_states += [[agent_state] + self.adjacent_locs(agent_state, self._field_size)]
				nxt_states_combs = []
				for comb in itertools.product(*nxt_agent_states):
					nxt_states_combs += [comb]
				nxt_state_prob = 1 / float(len(nxt_states_combs))
				if act == 'N':
					for nxt_state_tuple in nxt_states_combs:
						
						# store agent movements
						nxt_states = {''.join([str(x) for x in agent_states[0]]): [tuple(list(agent_states[0]) + [agent_levels[0]] + [1])]}
						for o_agent in nxt_state_tuple:
							o_agent_idx = nxt_state_tuple.index(o_agent) + 1
							loc_str = ''.join([str(x) for x in o_agent])
							if loc_str not in list(nxt_states.keys()):
								nxt_states[loc_str] = [tuple(list(agent_states[o_agent_idx]) + [agent_levels[o_agent_idx]] + [o_agent_idx])]
							else:
								nxt_states[loc_str] += [tuple(list(agent_states[o_agent_idx]) + [agent_levels[o_agent_idx]] + [o_agent_idx])]
						
						# verify collisions of agents)
						nxt_state_tmp = verify_collisions(nxt_states)
						
						# compose next state, collision free
						nxt_state = compose_state(nxt_state_tmp, fruit_pick)
						p[state_idx, states.index(nxt_state)] += nxt_state_prob
				
				elif act == 'U':
					for nxt_state_tuple in nxt_states_combs:
						
						# store agent movements
						nxt_pos = (max(agent_states[0][0] - 1, 0), agent_states[0][1])
						nxt_states = {''.join([str(x) for x in nxt_pos]): [tuple(list(agent_states[0]) + [agent_levels[0]] + [1])]}
						for o_agent in nxt_state_tuple:
							o_agent_idx = nxt_state_tuple.index(o_agent) + 1
							loc_str = ''.join([str(x) for x in o_agent])
							if loc_str not in list(nxt_states.keys()):
								nxt_states[loc_str] = [tuple(list(agent_states[o_agent_idx]) + [agent_levels[o_agent_idx]] + [o_agent_idx])]
							else:
								nxt_states[loc_str] += [tuple(list(agent_states[o_agent_idx]) + [agent_levels[o_agent_idx]] + [o_agent_idx])]
						
						# verify collisions of agents
						nxt_state_tmp = verify_collisions(nxt_states)
						
						# compose next state, collision free
						nxt_state = compose_state(nxt_state_tmp, fruit_pick)
						p[state_idx, states.index(nxt_state)] += nxt_state_prob
				
				elif act == 'D':
					for nxt_state_tuple in nxt_states_combs:
						
						# store agent movements
						nxt_pos = (min(agent_states[0][0] + 1, n_rows - 1), agent_states[0][1])
						nxt_states = {''.join([str(x) for x in nxt_pos]): [tuple(list(agent_states[0]) + [agent_levels[0]] + [1])]}
						for o_agent in nxt_state_tuple:
							o_agent_idx = nxt_state_tuple.index(o_agent) + 1
							loc_str = ''.join([str(x) for x in o_agent])
							if loc_str not in list(nxt_states.keys()):
								nxt_states[loc_str] = [tuple(list(agent_states[o_agent_idx]) + [agent_levels[o_agent_idx]] + [o_agent_idx])]
							else:
								nxt_states[loc_str] += [tuple(list(agent_states[o_agent_idx]) + [agent_levels[o_agent_idx]] + [o_agent_idx])]
						
						# verify collisions of agents
						nxt_state_tmp = verify_collisions(nxt_states)
						
						# compose next state, collision free
						nxt_state = compose_state(nxt_state_tmp, fruit_pick)
						p[state_idx, states.index(nxt_state)] += nxt_state_prob
				
				elif act == 'L':
					for nxt_state_tuple in nxt_states_combs:
						
						# store agent movements
						nxt_pos = (agent_states[0][0], max(agent_states[0][1] - 1, 0))
						nxt_states = {''.join([str(x) for x in nxt_pos]): [tuple(list(agent_states[0]) + [agent_levels[0]] + [1])]}
						for o_agent in nxt_state_tuple:
							o_agent_idx = nxt_state_tuple.index(o_agent) + 1
							loc_str = ''.join([str(x) for x in o_agent])
							if loc_str not in list(nxt_states.keys()):
								nxt_states[loc_str] = [tuple(list(agent_states[o_agent_idx]) + [agent_levels[o_agent_idx]] + [o_agent_idx])]
							else:
								nxt_states[loc_str] += [tuple(list(agent_states[o_agent_idx]) + [agent_levels[o_agent_idx]] + [o_agent_idx])]
						
						# verify collisions of agents
						nxt_state_tmp = verify_collisions(nxt_states)
						
						# compose next state, collision free
						nxt_state = compose_state(nxt_state_tmp, fruit_pick)
						p[state_idx, states.index(nxt_state)] += nxt_state_prob
				
				elif act == 'R':
					for nxt_state_tuple in nxt_states_combs:
						
						# store agent movements
						nxt_pos = (agent_states[0][0], min(agent_states[0][1] + 1, n_cols - 1))
						nxt_states = {''.join([str(x) for x in nxt_pos]): [tuple(list(agent_states[0]) + [agent_levels[0]] + [1])]}
						for o_agent in nxt_state_tuple:
							o_agent_idx = nxt_state_tuple.index(o_agent) + 1
							loc_str = ''.join([str(x) for x in o_agent])
							if loc_str not in list(nxt_states.keys()):
								nxt_states[loc_str] = [tuple(list(agent_states[o_agent_idx]) + [agent_levels[o_agent_idx]] + [o_agent_idx])]
							else:
								nxt_states[loc_str] += [tuple(list(agent_states[o_agent_idx]) + [agent_levels[o_agent_idx]] + [o_agent_idx])]
						
						# verify collisions of agents
						nxt_state_tmp = verify_collisions(nxt_states)
						
						# compose next state, collision free
						nxt_state = compose_state(nxt_state_tmp, fruit_pick)
						p[state_idx, states.index(nxt_state)] += nxt_state_prob
					
				elif act == 'Lo':
					agents = [(agent, level) for agent, level in zip(agent_states, agent_levels)]
					agent_adj_locs = self.adjacent_locs(agent_states[0], self._field_size)
					if food_loc in agent_adj_locs and food_loc != agent_states[0]:
						if food_lvl <= agent_levels[0]:
							for nxt_state_tuple in nxt_states_combs:
								nxt_pos = agent_states[0]
								
								# store agent movements
								nxt_states = {''.join([str(x) for x in nxt_pos]): [tuple(list(nxt_pos) + [agent_levels[0]] + [1])]}
								for o_agent in nxt_state_tuple:
									o_agent_idx = nxt_state_tuple.index(o_agent) + 1
									loc_str = ''.join([str(x) for x in o_agent])
									if loc_str not in list(nxt_states.keys()):
										nxt_states[loc_str] = [tuple(list(agent_states[o_agent_idx]) + [agent_levels[o_agent_idx]] + [o_agent_idx])]
									else:
										nxt_states[loc_str] += [tuple(list(agent_states[o_agent_idx]) + [agent_levels[o_agent_idx]] + [o_agent_idx])]
										
								# verify collisions of agents
								nxt_state_tmp = verify_collisions(nxt_states)
								
								# compose next state, collision free
								nxt_state = ''
								for state_tmp in nxt_state_tmp:
									nxt_state += ''.join([str(x) for x in state_tmp])
								nxt_state += '1'
								p[state_idx, states.index(nxt_state)] += nxt_state_prob
						else:
							load, load_agents = self.can_load(food_loc, food_lvl, agents)
							food_adj_locs = self.adjacent_locs(food_loc, self._field_size)
							if agent_states[0] in load_agents:
								load_agents.remove(agent_states[0])
							n_load_agents = int(len(load_agents))
							load_prob = (1 / 2.0**n_load_agents) * (1 / 5.0**(self._n_agents - n_load_agents - 1))
							
							if load:
								nxt_pos = agent_states[0]
								for nxt_state_tuple in nxt_states_combs:
									comb_lvl = agent_levels[0]
									nxt_states = {''.join([str(x) for x in nxt_pos]): [tuple(list(nxt_pos) + [agent_levels[0]] + [1])]}
									for o_agent in nxt_state_tuple:
										o_agent_idx = nxt_state_tuple.index(o_agent) + 1
										loc_str = ''.join([str(x) for x in o_agent])
										if o_agent in food_adj_locs and o_agent == agent_states[o_agent_idx]:
											comb_lvl += agent_levels[o_agent_idx]
										if loc_str not in list(nxt_states.keys()):
											nxt_states[loc_str] = [tuple(list(agent_states[o_agent_idx]) + [agent_levels[o_agent_idx]] + [o_agent_idx])]
										else:
											nxt_states[loc_str] += [tuple(list(agent_states[o_agent_idx]) + [agent_levels[o_agent_idx]] + [o_agent_idx])]
											
									# verify collisions of agents
									nxt_state_tmp = verify_collisions(nxt_states)
											
									# compose next state, collision free
									nxt_state = ''
									for state_tmp in nxt_state_tmp:
										nxt_state += ''.join([str(x) for x in state_tmp])
									
									if comb_lvl >= food_lvl:
										nxt_state += '1'
										p[state_idx, states.index(nxt_state)] += load_prob
									
									else:
										nxt_state += str(fruit_pick)
										p[state_idx, states.index(nxt_state)] += nxt_state_prob
							
							else:
								for nxt_state_tuple in nxt_states_combs:
									# store agent movements
									nxt_states = {''.join([str(x) for x in agent_states[0]]): [tuple(list(agent_states[0]) + [agent_levels[0]] + [1])]}
									for o_agent in nxt_state_tuple:
										o_agent_idx = nxt_state_tuple.index(o_agent) + 1
										loc_str = ''.join([str(x) for x in o_agent])
										if loc_str not in list(nxt_states.keys()):
											nxt_states[loc_str] = [tuple(list(agent_states[o_agent_idx]) + [agent_levels[o_agent_idx]] + [o_agent_idx])]
										else:
											nxt_states[loc_str] += [tuple(list(agent_states[o_agent_idx]) + [agent_levels[o_agent_idx]] + [o_agent_idx])]
									
									# verify collisions of agents
									nxt_state_tmp = verify_collisions(nxt_states)
									
									# compose next state, collision free
									nxt_state = compose_state(nxt_state_tmp, fruit_pick)
									p[state_idx, states.index(nxt_state)] += nxt_state_prob
										
				else:
					print(colored('Action not recognized. Skipping matrix probability', 'red'))
					continue
				
				if sum(p[state_idx]) > 0:
					p[state_idx] = p[state_idx] / sum(p[state_idx])
			
			P += [csr_matrix(p)]
			
		return P
	
	def generate_rewards(self, states: Tuple, actions: Tuple, food: Tuple) -> np.ndarray:
		nX = len(states)
		nA = len(actions)
		r = np.zeros((nX, nA))
		
		food_loc = food[:-1]
		food_lvl = food[-1]
		
		food_adj_locs = self.adjacent_locs(food_loc, self._field_size)
		
		for state in states:
			agents_locs, agents_lvl, food_state = self.get_state_tuple(state)
			if agents_locs[0] in food_adj_locs and food_state == 1:
				r[states.index(state), :] = 1.0
		
		return r
	
	def generate_world_food(self, food: Tuple) -> None:
		
		dict_idx = ''.join([str(x) for x in food[:-1]])
		
		if len(self._states) == 0:
			self._states = self.generate_states()
		
		if len(self._actions) == 0:
			self._actions = self.generate_actions()
		
		transitions = self.generate_transitions(self._states, self._actions, food)
		rewards = self.generate_rewards(self._states, self._actions, food)
		
		self._transitions[dict_idx] = transitions
		self._rewards[dict_idx] = rewards
	
	
class MultiAgentForagingPlan(ForagingPlan):
	
	def __init__(self, field_size: Tuple, max_food: int, max_food_level: int, n_agents: int, agent_level: List[int], min_food_level: int = 1):
		super().__init__(field_size, max_food, max_food_level, n_agents, agent_level, min_food_level)
	
	def generate_actions(self) -> Tuple:
		
		single_agent_actions = ['N', 'U', 'D', 'L', 'R', 'Lo']
		
		actions = []
		
		for comb in itertools.product(single_agent_actions, repeat=self._n_agents):
			actions += [comb]
		
		return tuple(actions)
		
	def generate_transitions(self, states: Tuple, actions: Tuple, food: Tuple) -> List[csr_matrix]:
		
		def verify_collisions(agents_states: Dict) -> List:
			nxt_state_list = []
			collisions = any([len(agents_states[loc]) > 1 for loc in agents_states.keys()])
			while collisions:
				states_keys = list(agents_states.keys())
				for loc in states_keys:
					if len(agents_states[loc]) > 1:
						for agent in agents_states[loc]:
							agents_states[loc].remove(agent)
							agent_orig = ''.join([str(x) for x in agent[0:2]])
							if agent_orig in list(agents_states.keys()):
								agents_states[agent_orig] += [agent]
							else:
								agents_states[agent_orig] = [agent]
				collisions = any([len(agents_states[loc]) > 1 for loc in agents_states.keys()])
			
			for loc in agents_states.keys():
				for agent in agents_states[loc]:
					if len(nxt_state_list) < agent[-1]:
						nxt_state_list = [[int(x) for x in loc] + [agents_states[loc][0][2]]]
					else:
						nxt_state_list = nxt_state_list[:agent[-1]] + [[int(x) for x in loc] + [agents_states[loc][0][2]]] + nxt_state_list[agent[-1]:]
			
			return nxt_state_list
		
		def compose_state(state_lst: List, fruit_got: int) -> str:
			comp_state = ''
			
			for s in state_lst:
				comp_state += ''.join([str(x) for x in s])
			
			comp_state += str(fruit_got)
			return comp_state
		
		nX = len(states)
		P = []
		n_rows, n_cols = self._field_size
		
		food_loc = food[:-1]
		food_lvl = food[-1]
		
		for joint_act in actions:
			
			agent_actions = list(joint_act)
			p = np.zeros((nX, nX))
			
			for state in states:
				
				state_idx = states.index(state)
				agent_states, agent_levels, food_pick = self.get_state_tuple(state)
				loading_lvls = []
				nxt_states = {}
				
				agent_idx = 0
				for act in agent_actions:
					
					agent_loc = agent_states[agent_idx]
					agent_lvl = agent_levels[agent_idx]
					
					if act == 'N':
						loc_str = ''.join([str(x) for x in agent_loc])
						if loc_str not in list(nxt_states.keys()):
							nxt_states[loc_str] = [tuple(list(agent_loc) + [agent_lvl] + [agent_idx])]
						else:
							nxt_states[loc_str] += [tuple(list(agent_loc) + [agent_lvl] + [agent_idx])]
							
					elif act == 'U':
						nxt_pos = (max(agent_loc[0] - 1, 0), agent_loc[1])
						if nxt_pos != food_loc:
							loc_str = ''.join([str(x) for x in nxt_pos])
						else:
							loc_str = ''.join([str(x) for x in agent_loc])
						if loc_str not in list(nxt_states.keys()):
							nxt_states[loc_str] = [tuple(list(agent_loc) + [agent_lvl] + [agent_idx])]
						else:
							nxt_states[loc_str] += [tuple(list(agent_loc) + [agent_lvl] + [agent_idx])]
					
					elif act == 'D':
						nxt_pos = (min(agent_loc[0] + 1, n_rows - 1), agent_loc[1])
						if nxt_pos != food_loc:
							loc_str = ''.join([str(x) for x in nxt_pos])
						else:
							loc_str = ''.join([str(x) for x in agent_loc])
						if loc_str not in list(nxt_states.keys()):
							nxt_states[loc_str] = [tuple(list(agent_loc) + [agent_lvl] + [agent_idx])]
						else:
							nxt_states[loc_str] += [tuple(list(agent_loc) + [agent_lvl] + [agent_idx])]
					
					elif act == 'L':
						nxt_pos = (agent_loc[0], max(agent_loc[1] - 1, 0))
						if nxt_pos != food_loc:
							loc_str = ''.join([str(x) for x in nxt_pos])
						else:
							loc_str = ''.join([str(x) for x in agent_loc])
						if loc_str not in list(nxt_states.keys()):
							nxt_states[loc_str] = [tuple(list(agent_loc) + [agent_lvl] + [agent_idx])]
						else:
							nxt_states[loc_str] += [tuple(list(agent_loc) + [agent_lvl] + [agent_idx])]
					
					elif act == 'R':
						nxt_pos = (agent_loc[0], min(agent_loc[1] + 1, n_cols - 1))
						if nxt_pos != food_loc:
							loc_str = ''.join([str(x) for x in nxt_pos])
						else:
							loc_str = ''.join([str(x) for x in agent_loc])
						if loc_str not in list(nxt_states.keys()):
							nxt_states[loc_str] = [tuple(list(agent_loc) + [agent_lvl] + [agent_idx])]
						else:
							nxt_states[loc_str] += [tuple(list(agent_loc) + [agent_lvl] + [agent_idx])]
					
					elif act == 'Lo':
						loc_str = ''.join([str(x) for x in agent_loc])
						if loc_str not in list(nxt_states.keys()):
							nxt_states[loc_str] = [tuple(list(agent_loc) + [agent_lvl] + [agent_idx])]
						else:
							nxt_states[loc_str] += [tuple(list(agent_loc) + [agent_lvl] + [agent_idx])]
					
						agent_adj_locs = self.adjacent_locs(agent_loc, self._field_size)
						if food_loc in agent_adj_locs:
							loading_lvls += [agent_lvl]
					
					else:
						print(colored('Action not recognized. Skipping matrix probability', 'red'))
						agent_idx += 1
						continue
					
					agent_idx += 1
				
				nxt_states_tmp = verify_collisions(nxt_states)
				
				# Update food pick state
				if len(loading_lvls) < 1:
					food_state = food_pick
				
				else:
					load_level = 0
					for agent_lvl in loading_lvls:
						load_level += agent_lvl
					
					food_state = 1 if load_level >= food_lvl else food_pick
				
				nxt_state = compose_state(nxt_states_tmp, food_state)
				p[state_idx, states.index(nxt_state)] = 1.0
				
			P += [csr_matrix(p)]
		
		return P
	
	def generate_rewards(self, states: Tuple, actions: Tuple, food: Tuple) -> np.ndarray:
		nX = len(states)
		nA = len(actions)
		r = np.zeros((nX, nA))
		
		food_loc = food[:-1]
		food_lvl = food[-1]
		
		for state in states:
			agents_locs, agents_lvl, food_state = self.get_state_tuple(state)
			agents = [(agent, level) for agent, level in zip(agents_locs, agents_lvl)]
			load, adj_agents = self.can_load(food_loc, food_lvl, agents)
			if load and food_state == 1:
				r[states.index(state), :] = 1.0
			elif len(adj_agents) > 0 and food_state == 0:
				r[states.index(state), :] = len(adj_agents) / self._n_agents * 0.1
		
		return r


class SingleAgentForagingPlan(ForagingPlan):
	
	def __init__(self, field_size: Tuple, max_food: int, max_food_level: int, n_agents: int, agent_level: List[int], optimal_pol_lst: List[np.ndarray],
				 joint_acts: Tuple, food_pos_lst: List[Tuple], social_roles: List = None, min_food_level: int = 1):
		
		self._joint_acts = joint_acts
		self._optimal_pol_lst = optimal_pol_lst
		self._food_pos_lst = food_pos_lst
		self._social_roles = social_roles
		super().__init__(field_size, max_food, max_food_level, n_agents, agent_level, min_food_level)
	
	@property
	def food_pos_lst(self) -> List[Tuple]:
		return self._food_pos_lst
	
	def generate_transitions(self, states: Tuple, actions: Tuple, food: Tuple) -> List[csr_matrix]:
		
		def verify_collisions(agents_states: Dict) -> List:
			nxt_state_list = []
			collisions = any([len(agents_states[loc]) > 1 for loc in agents_states.keys()])
			while collisions:
				states_keys = list(agents_states.keys())
				for loc in states_keys:
					if len(agents_states[loc]) > 1:
						for agent in agents_states[loc]:
							agents_states[loc].remove(agent)
							agent_orig = ''.join([str(x) for x in agent[0:2]])
							if agent_orig in list(agents_states.keys()):
								agents_states[agent_orig] += [agent]
							else:
								agents_states[agent_orig] = [agent]
				collisions = any([len(agents_states[loc]) > 1 for loc in agents_states.keys()])
			
			for loc in agents_states.keys():
				for agent in agents_states[loc]:
					if len(nxt_state_list) < agent[-1]:
						nxt_state_list = [[int(x) for x in loc] + [agents_states[loc][0][2]]]
					else:
						nxt_state_list = nxt_state_list[:agent[-1]] + [[int(x) for x in loc] + [agents_states[loc][0][2]]] + nxt_state_list[agent[-1]:]
			
			return nxt_state_list
		
		def compose_state(state_lst: List, fruit_got: int) -> str:
			comp_state = ''
			
			for s in state_lst:
				comp_state += ''.join([str(x) for x in s])
			
			comp_state += str(fruit_got)
			return comp_state
		
		nX = len(states)
		P = []
		n_rows, n_cols = self._field_size
		
		food_loc = food[:-1]
		food_lvl = food[-1]
		optimal_pol = self._optimal_pol_lst[self._food_pos_lst.index(food_loc)]
		
		for agent_act in actions:
			
			p = np.zeros((nX, nX))
			
			for state in states:
				
				state_idx = states.index(state)
				opt_pol_act = np.argwhere(optimal_pol[state_idx] == optimal_pol[state_idx].max()).ravel()
				# opt_pol_act = [np.argmax(optimal_pol[state_idx])]
				agent_states, agent_levels, food_pick = self.get_state_tuple(state)
				
				for opt_act_idx in opt_pol_act:
					
					joint_act = [agent_act] + list(self._joint_acts[opt_act_idx][1:])
					loading_lvls = []
					nxt_states = {}
					
					agent_idx = 0
					for act in joint_act:
						
						agent_loc = agent_states[agent_idx]
						agent_lvl = agent_levels[agent_idx]
					
						if act == 'N':
							loc_str = ''.join([str(x) for x in agent_loc])
							if loc_str not in list(nxt_states.keys()):
								nxt_states[loc_str] = [tuple(list(agent_loc) + [agent_lvl] + [agent_idx])]
							else:
								nxt_states[loc_str] += [tuple(list(agent_loc) + [agent_lvl] + [agent_idx])]
								
						elif act == 'U':
							nxt_pos = (max(agent_loc[0] - 1, 0), agent_loc[1])
							if nxt_pos == food_loc:
								nxt_pos = agent_loc
							loc_str = ''.join([str(x) for x in nxt_pos])
							if loc_str not in list(nxt_states.keys()):
								nxt_states[loc_str] = [tuple(list(agent_loc) + [agent_lvl] + [agent_idx])]
							else:
								nxt_states[loc_str] += [tuple(list(agent_loc) + [agent_lvl] + [agent_idx])]
						
						elif act == 'D':
							nxt_pos = (min(agent_loc[0] + 1, n_rows - 1), agent_loc[1])
							if nxt_pos == food_loc:
								nxt_pos = agent_loc
							loc_str = ''.join([str(x) for x in nxt_pos])
							if loc_str not in list(nxt_states.keys()):
								nxt_states[loc_str] = [tuple(list(agent_loc) + [agent_lvl] + [agent_idx])]
							else:
								nxt_states[loc_str] += [tuple(list(agent_loc) + [agent_lvl] + [agent_idx])]
						
						elif act == 'L':
							nxt_pos = (agent_loc[0], max(agent_loc[1] - 1, 0))
							if nxt_pos == food_loc:
								nxt_pos = agent_loc
							loc_str = ''.join([str(x) for x in nxt_pos])
							if loc_str not in list(nxt_states.keys()):
								nxt_states[loc_str] = [tuple(list(agent_loc) + [agent_lvl] + [agent_idx])]
							else:
								nxt_states[loc_str] += [tuple(list(agent_loc) + [agent_lvl] + [agent_idx])]
						
						elif act == 'R':
							nxt_pos = (agent_loc[0], min(agent_loc[1] + 1, n_cols - 1))
							if nxt_pos == food_loc:
								nxt_pos = agent_loc
							loc_str = ''.join([str(x) for x in nxt_pos])
							if loc_str not in list(nxt_states.keys()):
								nxt_states[loc_str] = [tuple(list(agent_loc) + [agent_lvl] + [agent_idx])]
							else:
								nxt_states[loc_str] += [tuple(list(agent_loc) + [agent_lvl] + [agent_idx])]
						
						elif act == 'Lo':
							loc_str = ''.join([str(x) for x in agent_loc])
							if loc_str not in list(nxt_states.keys()):
								nxt_states[loc_str] = [tuple(list(agent_loc) + [agent_lvl] + [agent_idx])]
							else:
								nxt_states[loc_str] += [tuple(list(agent_loc) + [agent_lvl] + [agent_idx])]
						
							agent_adj_locs = self.adjacent_locs(agent_loc, self._field_size)
							if food_loc in agent_adj_locs:
								loading_lvls += [agent_lvl]
						
						else:
							print(colored('Action not recognized. Skipping matrix probability', 'red'))
							agent_idx += 1
							continue
					
						agent_idx += 1
					
					nxt_states_tmp = verify_collisions(nxt_states)
					
					# Update food pick state
					if len(loading_lvls) < 1:
						food_state = food_pick
					
					else:
						load_level = 0
						for agent_lvl in loading_lvls:
							load_level += agent_lvl
						
						food_state = 1 if load_level >= food_lvl else food_pick
					
					nxt_state = compose_state(nxt_states_tmp, food_state)
					p[state_idx, states.index(nxt_state)] += optimal_pol[state_idx, opt_act_idx]
					# p[state_idx, states.index(nxt_state)] += 1.0
				
				# p[state_idx] = p[state_idx] / sum(p[state_idx])
			
			P += [csr_matrix(p)]
		
		return P
	
	def generate_rewards(self, states: Tuple, actions: Tuple, food: Tuple, social_roles: List = None) -> np.ndarray:
		
		nX = len(states)
		nA = len(actions)
		r = np.zeros((nX, nA))
		
		food_loc = food[:-1]
		food_lvl = food[-1]
		
		social_locs = []
		if not social_roles:
			social_locs = self.adjacent_locs(food_loc, self._field_size)
			while food_loc in social_locs:
				social_locs.remove(food_loc)
		
		else:
			field_rows, field_cols = self._field_size
			food_row, food_col = food_loc
			for role in social_roles:
				if role == 'up':
					social_loc = (max(food_row - 1, 0), food_col)
				elif role == 'down':
					social_loc = (min(food_row + 1, field_rows - 1), food_col)
				elif role == 'left':
					social_loc = (food_row, max(food_col - 1, 0))
				elif role == 'right':
					social_loc = (food_row, min(food_col + 1, field_cols - 1))
				else:
					social_loc = (food_row, food_col)
					
				if social_loc != food_loc:
					social_locs += [social_loc]
		
		for state in states:
			agents_locs, agents_lvl, food_state = self.get_state_tuple(state)
			if len(social_locs) > 0 and agents_locs[0] in social_locs:
				agents = [(agent, level) for agent, level in zip(agents_locs, agents_lvl)]
				load, adj_agents = self.can_load(food_loc, food_lvl, agents)
				if load and food_state == 1:
					r[states.index(state), :] = 1.0
				elif agents_locs[0] in adj_agents and food_state == 0:
					r[states.index(state), :] = 0.1
		
		return r
	
	def generate_world_food(self, food: Tuple) -> None:
		dict_idx = ''.join([str(x) for x in food[:-1]])
		
		if len(self._states) == 0:
			self._states = self.generate_states()
		
		if len(self._actions) == 0:
			self._actions = self.generate_actions()
		
		transitions = self.generate_transitions(self._states, self._actions, food)
		rewards = self.generate_rewards(self._states, self._actions, food, self._social_roles)
		
		self._transitions[dict_idx] = transitions
		self._rewards[dict_idx] = rewards
		
	