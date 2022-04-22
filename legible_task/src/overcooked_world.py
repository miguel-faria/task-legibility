#! /usr/bin/env python
import time

import numpy as np
import re
from termcolor import colored
from abc import ABC
from itertools import combinations, permutations
from typing import List, Dict, Tuple


##################################
#### Overcooked Burger Worlds ####
##################################
class BurgerOvercooked(object):

	def generate_states(self, n_rows: int, n_cols: int, wall_states: List[Tuple[int, int]], n_plates: int = 2) -> Tuple[np.ndarray, List[Tuple]]:
		
		rng_gen = np.random.default_rng(time.time())
		states = []
		obj_loc = []
		objs = ['meat', 'veggies', 'bread', 'fry-pan', 'knife', 'plate', 'deliver']
		pick_objs = ['meat', 'veggies', 'bread', 'dice-meat', 'dice-veggies']
		plates_loc = []
		wall_states_cp = np.array(wall_states.copy())
		n_objs = len(objs)
		n_pick_objs = len(pick_objs)
		for obj in objs:
			loc = rng_gen.choice(wall_states_cp)
			if ((max(loc[0] - 1, 1), loc[1]) not in wall_states or (min(loc[0] + 1, n_rows + 1), loc[1]) not in wall_states or
					(loc[0], max(loc[1] - 1, 1)) not in wall_states or (loc[0], min(loc[1] + 1, n_cols + 1)) not in wall_states):
				obj_loc += [(loc[0], loc[1], obj)]
				np.delete(wall_states_cp, loc)
		for i in range(n_rows):
			for j in range(n_cols):
				cur_loc = (i + 1, j + 1)
				if cur_loc not in wall_states:
					states += [(cur_loc[0], cur_loc[1], 'N', 'N', 'N')]
					for obj in pick_objs:
						states += [(cur_loc[0], cur_loc[1], obj, 'N', 'N')]
					for n in range(1, n_objs + 1):
						combs = combinations(objs, n)
						for comb in combs:
							if cur_loc in obj_loc and str(comb).find(objs[obj_loc.index(cur_loc)]) == -1:
								continue
							states += [''.join(str(x) + ' ' for x in cur_loc) + ''.join(comb)]
		
		return np.array(states), obj_loc
	
	def generate_actions(self) -> np.ndarray:
		return np.array(['Up', 'Down', 'Left', 'Right', 'Pick', 'Put', 'Cook', 'Plate', 'Dice', 'Deliver'])
	
	def generate_stochastic_probabilities(self, states, actions, obj_states, max_rows, max_cols, fail_chance) -> Dict[str, np.ndarray]:
		
		objs_loc = []
		objs = []
		for x, y, o in obj_states:
			objs_loc += [(x, y)]
			objs += [o]
		n_objs = len(objs)
		nX = len(states)
		P = {}
		state_lst = list(states)
		
		for a in actions:
			p = np.zeros((nX, nX))
			if a == 'Up':
				for state in states:
					state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
					curr_row = int(state_split.group(1))
					curr_col = int(state_split.group(2))
					curr_state_obj = state_split.group(3)
					state_idx = state_lst.index(state)
					tmp_nxt_row = min(max_rows, curr_row + 1)
					next_loc = (tmp_nxt_row, curr_col)
					nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj
					
					if nxt_state in states:
						nxt_idx = state_lst.index(nxt_state)
					
					else:
						obj_loc_idx = objs_loc.index(next_loc)
						if curr_state_obj == 'N':
							nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])
						
						else:
							nxt_obj_lst = list(curr_state_obj) + list(objs[obj_loc_idx])
							obj_perm = [''.join(x) for x in list(permutations(nxt_obj_lst))]
							nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[0]
							for i in range(1, len(obj_perm)):
								if nxt_state in states:
									break
								nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[i]
							nxt_idx = state_lst.index(nxt_state)
					
					if state_idx != nxt_idx:
						p[state_idx][nxt_idx] = 1.0 - fail_chance
						p[state_idx][state_idx] = fail_chance
					
					else:
						p[state_idx][state_idx] = 1.0
			
			elif a == 'Down':
				for state in states:
					state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
					curr_row = int(state_split.group(1))
					curr_col = int(state_split.group(2))
					curr_state_obj = state_split.group(3)
					state_idx = state_lst.index(state)
					tmp_nxt_row = max(1, curr_row - 1)
					next_loc = (tmp_nxt_row, curr_col)
					nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj
					
					if nxt_state in states:
						nxt_idx = state_lst.index(nxt_state)
					
					else:
						obj_loc_idx = objs_loc.index(next_loc)
						if curr_state_obj == 'N':
							nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])
						
						else:
							nxt_obj_lst = list(curr_state_obj) + list(objs[obj_loc_idx])
							obj_perm = [''.join(x) for x in list(permutations(nxt_obj_lst))]
							nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[0]
							for i in range(1, len(obj_perm)):
								if nxt_state in states:
									break
								nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[i]
							nxt_idx = state_lst.index(nxt_state)
					
					if state_idx != nxt_idx:
						p[state_idx][nxt_idx] = 1.0 - fail_chance
						p[state_idx][state_idx] = fail_chance
					
					else:
						p[state_idx][state_idx] = 1.0
			
			elif a == 'Left':
				for state in states:
					state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
					curr_row = int(state_split.group(1))
					curr_col = int(state_split.group(2))
					curr_state_obj = state_split.group(3)
					state_idx = state_lst.index(state)
					tmp_nxt_col = max(1, curr_col - 1)
					next_loc = (curr_row, tmp_nxt_col)
					nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj
					
					if nxt_state in states:
						nxt_idx = state_lst.index(nxt_state)
					
					else:
						obj_loc_idx = objs_loc.index(next_loc)
						if curr_state_obj == 'N':
							nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])
						
						else:
							nxt_obj_lst = list(curr_state_obj) + list(objs[obj_loc_idx])
							obj_perm = [''.join(x) for x in list(permutations(nxt_obj_lst))]
							nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[0]
							for i in range(1, len(obj_perm)):
								if nxt_state in states:
									break
								nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[i]
							nxt_idx = state_lst.index(nxt_state)
					
					if state_idx != nxt_idx:
						p[state_idx][nxt_idx] = 1.0 - fail_chance
						p[state_idx][state_idx] = fail_chance
					
					else:
						p[state_idx][state_idx] = 1.0
			
			elif a == 'Right':
				for state in states:
					state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
					curr_row = int(state_split.group(1))
					curr_col = int(state_split.group(2))
					curr_state_obj = state_split.group(3)
					state_idx = state_lst.index(state)
					tmp_nxt_col = min(max_cols, curr_col + 1)
					next_loc = (curr_row, tmp_nxt_col)
					nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj
					if nxt_state in states:
						nxt_idx = state_lst.index(nxt_state)
					
					else:
						obj_loc_idx = objs_loc.index(next_loc)
						if curr_state_obj == 'N':
							nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])
						
						else:
							nxt_obj_lst = list(curr_state_obj) + list(objs[obj_loc_idx])
							obj_perm = [''.join(x) for x in list(permutations(nxt_obj_lst))]
							nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[0]
							for i in range(1, len(obj_perm)):
								if nxt_state in states:
									break
								nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[i]
							nxt_idx = state_lst.index(nxt_state)
					
					if state_idx != nxt_idx:
						p[state_idx][nxt_idx] = 1.0 - fail_chance
						p[state_idx][state_idx] = fail_chance
					
					else:
						p[state_idx][state_idx] = 1.0
			
			elif a == 'N':
				p = np.eye(nX)
			
			else:
				print(colored('Action not recognized. Skipping matrix probability', 'red'))
				continue
			
			P[a] = p
			
		return P
	
	@staticmethod
	def generate_rewards(goal, states, actions) -> np.ndarray:
		nX = len(states)
		nA = len(actions)
		
		c = np.zeros((nX, nA))
		
		for state in states:
			if state.find(goal) != -1:
				c[list(states).index(state), :] = 1.0
		
		return c
	
	@staticmethod
	def generate_costs(goal, states, actions) -> np.ndarray:
		nX = len(states)
		nA = len(actions)
		
		c = np.ones((nX, nA))
		
		for state in states:
			if state.find(goal) != -1:
				c[list(states).index(state), :] = 0.0
		
		return c
	
	def generate_world(self, n_rows, n_cols, obj_states, utility_func, goal, fail_chance=0.0) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
		states = self.generate_states(n_rows, n_cols, obj_states)
		actions = self.generate_actions()
		probabilities = self.generate_stochastic_probabilities(states, actions, obj_states, n_rows, n_cols, fail_chance)
		
		if utility_func.find('reward') != -1:
			utility = BurgerOvercooked.generate_rewards(goal, states, actions)
		else:
			utility = BurgerOvercooked.generate_costs(goal, states, actions)
		
		return states, actions, probabilities, utility
	
