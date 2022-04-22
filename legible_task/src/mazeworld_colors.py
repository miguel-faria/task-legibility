#! /usr/bin/env python
import itertools

import numpy as np
from termcolor import colored
from typing import List, Tuple, Dict
from src.mdpworld import MDPWorld


#########################
#### Base Word World ####
#########################
class RobotExecuteColorForceSeqGame(MDPWorld):
	
	def __init__(self, n_rows: int, n_cols: int, colors: List[str], color_states: List[Tuple], walls: List[Tuple], n_locs: int,
				 fail_chance: float = 0.0, seq_length: int = 4):
		self._n_rows = n_rows
		self._n_cols = n_cols
		self._colors = colors
		self._color_states = color_states
		self._walls = walls
		self._fail_chance = fail_chance
		self._seq_length = seq_length
		self._n_valid_locs = n_locs
		
		self._states = []
		self._actions = []
	
	@staticmethod
	def get_state_str(state_tuple: Tuple) -> str:
		return ', '.join(' '.join(str(x) for x in elem) for elem in state_tuple)
	
	@staticmethod
	def get_state_tuple(state_str: str) -> Tuple:
		state = []
		state_split = state_str.split(', ')
		for elem in state_split:
			try:
				state += [tuple(map(int, elem.split(' ')))]
			except ValueError:
				state += [tuple(elem.split(' '))]
		
		return tuple(state)
	
	@staticmethod
	def get_action_tuple(action_str: str) -> Tuple[str]:
		return tuple(action_str.split(' '))
	
	def wall_exists(self, state: Tuple[int, int], action: str) -> bool:
		state_row = state[0]
		state_col = state[1]
		
		if action == 'U':
			up_move = (min(self._n_rows, state_row + 1), state_col)
			if up_move in self._walls:
				return True
		
		elif action == 'D':
			down_move = (max(0, state_row - 1), state_col)
			if down_move in self._walls:
				return True
		
		elif action == 'L':
			left_move = (state_row, max(0, state_col - 1))
			if left_move in self._walls:
				return True
		
		elif action == 'R':
			right_move = (state_row, min(self._n_cols, state_col + 1))
			if right_move in self._walls:
				return True
		
		else:
			return False
		
		return False
	
	def generate_states(self) -> np.ndarray:
		states = []
		locs = []
		objs = ['N']
		for _, _, o in self._color_states:
			objs += [o]
		upload_status = ['0', '1']
		for i in range(self._n_rows):
			for j in range(self._n_cols):
				curr_loc = (i + 1, j + 1)
				if curr_loc not in self._walls:
					locs += [' '.join(str(x) for x in curr_loc)]
		
		for loc in locs:
			for obj in objs:
				for status in upload_status:
					states += [', '.join((loc, obj, status))]
		
		return np.array(states, dtype=tuple)
	
	def generate_actions(self) -> np.ndarray:
		actions = ['U', 'D', 'L', 'R', 'N', 'Lo']
		return np.array(actions)
	
	def generate_stochastic_probabilities(self, states: np.ndarray, actions: np.ndarray) -> Dict[str, np.ndarray]:
		nX = len(states)
		state_lst = list(states)
		
		objs_loc = []
		objs_list = []
		for x, y, o in self._color_states:
			objs_loc += [(x, y)]
			objs_list += [o]
		
		P = {}
		
		for act in actions:
			p = np.zeros((nX, nX))
			for state in states:
				state_tuple = RobotExecuteColorSeqGame.get_state_tuple(state)
				state_idx = state_lst.index(state)
				nxt_state = ()
				curr_rob_loc = state_tuple[0]
				curr_state_color = state_tuple[1]
				curr_seq_status = state_tuple[2]
				if act == 'U':
					nxt_loc = (min(self._n_rows, curr_rob_loc[0] + 1), curr_rob_loc[1])
					if self.wall_exists(curr_rob_loc, act):
						nxt_state += (curr_rob_loc,)
					else:
						nxt_state += (nxt_loc,)
					
					nxt_state += (curr_state_color, curr_seq_status, )
				
				elif act == 'D':
					nxt_loc = (max(1, curr_rob_loc[0] - 1), curr_rob_loc[1])
					if self.wall_exists(curr_rob_loc, act):
						nxt_state += (curr_rob_loc,)
					else:
						nxt_state += (nxt_loc,)
						
					nxt_state += (curr_state_color, curr_seq_status, )
				
				elif act == 'L':
					nxt_loc = (curr_rob_loc[0], max(1, curr_rob_loc[1] - 1))
					if self.wall_exists(curr_rob_loc, act):
						nxt_state += (curr_rob_loc,)
					else:
						nxt_state += (nxt_loc,)
						
					nxt_state += (curr_state_color, curr_seq_status,)
				
				elif act == 'R':
					nxt_loc = (curr_rob_loc[0], min(self._n_cols, curr_rob_loc[1] + 1))
					if self.wall_exists(curr_rob_loc, act):
						nxt_state += (curr_rob_loc,)
					else:
						nxt_state += (nxt_loc,)
					
					nxt_state += (curr_state_color, curr_seq_status,)
				
				elif act == 'N':
					nxt_state = state_tuple
				
				elif act == 'Lo':
					nxt_state += (curr_rob_loc,)
					
					if curr_rob_loc in objs_loc:
						nxt_state += ((objs_list[objs_loc.index(curr_rob_loc)],),)
					else:
						nxt_state += (curr_state_color,)
					
					nxt_state += (curr_seq_status,)
					
				else:
					print(colored('Action not recognized. Skipping matrix probability', 'red'))
					continue
				
				nxt_idx = state_lst.index(RobotExecuteColorSeqGame.get_state_str(nxt_state))
				if curr_seq_status[0] == 0:
					seq_change_prob = 1 / self._n_valid_locs
					nxt_state_op = nxt_state[:-1] + (('1', ),)
					p[state_idx][state_lst.index(RobotExecuteColorSeqGame.get_state_str(nxt_state_op))] = seq_change_prob
					p[state_idx][nxt_idx] += (1.0 - self._fail_chance - seq_change_prob)
				else:
					p[state_idx][nxt_idx] += (1.0 - self._fail_chance)
				p[state_idx][state_idx] += self._fail_chance
			
			P[act] = p
		
		return P
	
	@staticmethod
	def generate_rewards(goal_states: List[int], states: np.ndarray, actions: np.ndarray) -> np.ndarray:
		nX = len(states)
		nA = len(actions)
		
		c = np.zeros((nX, nA))
		c[:, list(actions).index('Lo')] = -50.0
		for state in goal_states:
			if states[state].find('0') == -1:
				c[state, :] = 100.0
			else:
				c[state, :] = -100.0
		
		return c
	
	@staticmethod
	def generate_costs(goal_states: List[int], states: np.ndarray, actions: np.ndarray) -> np.ndarray:
		nX = len(states)
		nA = len(actions)
		
		c = np.ones((nX, nA))
		for state in goal_states:
			c[state, :] = 0.0
		
		return c
	
	def generate_world(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
		print('### Generating Word Maze World for Robot ###')
		print('Generating States')
		states = self.generate_states()
		print('Generating Actions')
		actions = self.generate_actions()
		print('Generating Transitions')
		probabilities = self.generate_stochastic_probabilities(states, actions)
		
		self._states = states
		self._actions = actions
		
		print('World Created')
		return states, actions, probabilities


class RobotExecuteColorSeqGame(MDPWorld):
	
	def __init__(self, n_rows: int, n_cols: int, colors: List[str], color_states: List[Tuple], walls: List[Tuple],
				 fail_chance: float = 0.0, seq_length: int = 4):
		self._n_rows = n_rows
		self._n_cols = n_cols
		self._colors = colors
		self._color_states = color_states
		self._walls = walls
		self._fail_chance = fail_chance
		self._seq_length = seq_length
		
		self._states = []
		self._actions = []
	
	@staticmethod
	def get_state_str(state_tuple: Tuple) -> str:
		return ', '.join(' '.join(str(x) for x in elem) for elem in state_tuple)
	
	@staticmethod
	def get_state_tuple(state_str: str) -> Tuple:
		state = []
		state_split = state_str.split(', ')
		for elem in state_split:
			try:
				state += [tuple(map(int, elem.split(' ')))]
			except ValueError:
				state += [tuple(elem.split(' '))]
		
		return tuple(state)
	
	@staticmethod
	def get_action_tuple(action_str: str) -> Tuple[str]:
		return tuple(action_str.split(' '))
	
	def wall_exists(self, state: Tuple[int, int], action: str) -> bool:
		state_row = state[0]
		state_col = state[1]
		
		if action == 'U':
			up_move = (min(self._n_rows, state_row + 1), state_col)
			if up_move in self._walls:
				return True
		
		elif action == 'D':
			down_move = (max(0, state_row - 1), state_col)
			if down_move in self._walls:
				return True
		
		elif action == 'L':
			left_move = (state_row, max(0, state_col - 1))
			if left_move in self._walls:
				return True
		
		elif action == 'R':
			right_move = (state_row, min(self._n_cols, state_col + 1))
			if right_move in self._walls:
				return True
		
		else:
			return False
		
		return False
	
	def generate_states(self) -> np.ndarray:
		states = []
		locs = []
		objs = ['N']
		for x, y, color in self._color_states:
			objs += [color]
		for i in range(self._n_rows):
			for j in range(self._n_cols):
				curr_loc = (i + 1, j + 1)
				if curr_loc not in self._walls:
					locs += [' '.join(str(x) for x in curr_loc)]
		
		for loc in locs:
			for obj in objs:
				if obj != 'N':
					for i in range(self._seq_length):
						states += [', '.join((loc, obj + str(i + 1)))]
				else:
					states += [', '.join((loc, obj))]
		
		return np.array(states, dtype=tuple)
	
	def generate_actions(self) -> np.ndarray:
		actions = ['U', 'D', 'L', 'R', 'N']
		for i in range(self._seq_length):
			actions += ['Lo' + str(i + 1)]
		return np.array(actions)
	
	def generate_stochastic_probabilities(self, states: np.ndarray, actions: np.ndarray) -> Dict[str, np.ndarray]:
		nX = len(states)
		state_lst = list(states)
		
		objs_loc = []
		objs_list = []
		for x, y, o in self._color_states:
			objs_loc += [(x, y)]
			objs_list += [o]
		
		P = {}
		
		for act in actions:
			p = np.zeros((nX, nX))
			for state in states:
				state_tuple = RobotExecuteColorSeqGame.get_state_tuple(state)
				state_idx = state_lst.index(state)
				nxt_state = ()
				nxt_state_color = ()
				curr_rob_loc = state_tuple[0]
				curr_state_color = state_tuple[-1][0]
				if act == 'U':
					nxt_state_color += (curr_state_color,)
					
					nxt_loc = (min(self._n_rows, curr_rob_loc[0] + 1), curr_rob_loc[1])
					if self.wall_exists(curr_rob_loc, act):
						nxt_state += (curr_rob_loc, )
					else:
						nxt_state += (nxt_loc, )
				
				elif act == 'D':
					nxt_state_color += (curr_state_color,)
					
					nxt_loc = (max(1, curr_rob_loc[0] - 1), curr_rob_loc[1])
					if self.wall_exists(curr_rob_loc, act):
						nxt_state += (curr_rob_loc, )
					else:
						nxt_state += (nxt_loc, )
				
				elif act == 'L':
					nxt_state_color += (curr_state_color,)
					
					nxt_loc = (curr_rob_loc[0], max(1, curr_rob_loc[1] - 1))
					if self.wall_exists(curr_rob_loc, act):
						nxt_state += (curr_rob_loc, )
					else:
						nxt_state += (nxt_loc, )
				
				elif act == 'R':
					nxt_state_color += (curr_state_color,)
					
					nxt_loc = (curr_rob_loc[0], min(self._n_cols, curr_rob_loc[1] + 1))
					if self.wall_exists(curr_rob_loc, act):
						nxt_state += (curr_rob_loc, )
					else:
						nxt_state += (nxt_loc, )
				
				elif act == 'N':
					nxt_state_color += (curr_state_color, )
					nxt_state += (curr_rob_loc, )
				
				elif act.find('Lo') != -1:
					if curr_rob_loc in objs_loc:
						nxt_state_color += (objs_list[objs_loc.index(curr_rob_loc)] + act[-1], )
					else:
						nxt_state_color += (curr_state_color, )
					
					nxt_state += (curr_rob_loc, )
				
				else:
					print(colored('Action not recognized. Skipping matrix probability', 'red'))
					continue
				
				nxt_state += (nxt_state_color, )
				nxt_idx = state_lst.index(RobotExecuteColorSeqGame.get_state_str(nxt_state))
				p[state_idx][nxt_idx] += (1.0 - self._fail_chance)
				p[state_idx][state_idx] += self._fail_chance
			
			P[act] = p
		
		return P
	
	@staticmethod
	def generate_rewards(goal_states: List[int], states: np.ndarray, actions: np.ndarray) -> np.ndarray:
		nX = len(states)
		nA = len(actions)
		
		c = np.zeros((nX, nA))
		for state in goal_states:
			c[state, :] = 100.0
		
		return c
	
	@staticmethod
	def generate_costs(goal_states: List[int], states: np.ndarray, actions: np.ndarray) -> np.ndarray:
		nX = len(states)
		nA = len(actions)
		
		c = np.ones((nX, nA))
		for state in goal_states:
			c[state, :] = 0.0
		
		return c
	
	def generate_world(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
		print('### Generating Word Maze World for Robot ###')
		print('Generating States')
		states = self.generate_states()
		print('Generating Actions')
		actions = self.generate_actions()
		print('Generating Transitions')
		probabilities = self.generate_stochastic_probabilities(states, actions)
		
		self._states = states
		self._actions = actions
		
		print('World Created')
		return states, actions, probabilities


class RobotTeamColorForceSeqGame(MDPWorld):
	
	def __init__(self, colors: List[str], start_states: List[Tuple], color_states: List[Tuple], walls: List[Tuple], color_seqs: List[Tuple],
				 n_locs: int, fail_chance: float = 0.0, n_robots: int = 2, seq_length: int = 4):
		self._colors = colors
		self._start_states = start_states
		self._color_states = color_states
		self._color_seqs = color_seqs
		self._walls = walls
		self._fail_chance = fail_chance
		self._n_robots = n_robots
		self._n_valid_locs = n_locs
		self._seq_length = seq_length
		
		self._states = []
		self._actions = []
	
	@staticmethod
	def get_state_str(state_tuple: Tuple) -> str:
		return ', '.join(' '.join(str(x) for x in elem) for elem in state_tuple)
	
	@staticmethod
	def get_state_tuple(state_str: str) -> Tuple:
		state = []
		state_split = state_str.split(', ')
		for elem in state_split:
			elem_split = elem.split(' ')
			elem_tuple = []
			for elem_2 in elem_split:
				try:
					elem_tuple += [int(elem_2)]
				except ValueError:
					elem_tuple += [elem_2]
			state += [tuple(elem_tuple)]
		
		return tuple(state)
	
	@staticmethod
	def get_action_tuple(action_str: str) -> Tuple[str]:
		return tuple(action_str.split(' '))
	
	def generate_states(self) -> np.ndarray:
		robot_states = ['F']
		for letter in self._colors:
			robot_states += [letter]
		states = []
		words = ['']
		
		for i in range(self._seq_length):
			for p in itertools.permutations(self._colors, (i + 1)):
				words += [''.join(p)]
			
		for state in itertools.product(robot_states, repeat=self._n_robots):
			for word in words:
				states += [', '.join(state + (''.join(word),))]
		
		return np.array(states)
	
	def generate_actions(self) -> np.ndarray:
		possible_actions = ['N']
		for letter in self._colors:
			for i in range(self._n_robots):
				possible_actions += ['U' + str(i + 1) + '_' + letter]
		return np.array(possible_actions, dtype=tuple)
	
	def action_detailed(self, action: str) -> Tuple:
		if action == 'N':
			return 'N', 'self'
		
		elif action.find('U') != -1:
			return 'U', action[-1:], int(action[1])
		
		else:
			print(colored('Unrecognized action for Team definition of a Word Maze world', 'red'))
			return 'N', 'self'
	
	def generate_stochastic_probabilities(self, states: np.ndarray, actions: np.ndarray) -> Dict[str, np.ndarray]:
		nX = len(states)
		state_lst = list(states)
		
		letter_states = {}
		
		for x, y, l in self._color_states:
			letter_states[l] = (x, y)
		
		P = {}
		
		for act in actions:
			p = np.zeros((nX, nX))
			for state in states:
				curr_state = RobotTeamColorSeqGame.get_state_tuple(state)
				state_idx = state_lst.index(state)
				robot_states = curr_state[:-1]
				seq_state = curr_state[-1][0]
				if act == 'N':
					state_transitions = np.zeros(nX)
					
					if np.all([curr_state[i] == 'F' for i in range(self._n_robots)]) or len(seq_state) >= self._seq_length:
						state_transitions[state_idx] = 1.0
					
					else:
						success_chance = 1 - self._fail_chance
						seq_change_prob = 1 / self._n_valid_locs
						
						use_robots = [i for i, x in enumerate(robot_states) if x[0] != 'F']
						if len(use_robots) > 0:
							for i in range(len(use_robots), 0, -1):
								use_robots_comb = itertools.combinations(use_robots, i)
								for comb in use_robots_comb:
									nxt_r_state = ()
									nxt_seq_states = []
									trans_prob = seq_change_prob ** max(1, len(comb))
									for idx in range(self._n_robots):
										if idx in comb:
											nxt_r_state += (('F',),)
										else:
											if any([robot_states[idx] == robot_states[c_idx] for c_idx in comb]):
												nxt_r_state += (('F',),)
											else:
												nxt_r_state += (robot_states[idx],)
									for perm in itertools.permutations(comb, len(comb)):
										nxt_seq_state = seq_state
										for p_idx in range(len(perm)):
											if nxt_seq_state.find(robot_states[perm[p_idx]][0]) == -1:
												nxt_seq_state += robot_states[perm[p_idx]][0]
										nxt_seq_states += [nxt_seq_state]
									for nxt_seq in nxt_seq_states:
										nxt_state = (nxt_r_state + ((nxt_seq[:self._seq_length],),))
										nxt_idx = state_lst.index(RobotTeamColorSeqGame.get_state_str(nxt_state))
										state_transitions[nxt_idx] += trans_prob
										success_chance -= trans_prob
						
							state_transitions[state_idx] += success_chance + self._fail_chance
							state_transitions = state_transitions / state_transitions.sum()
					
					p[state_idx] = state_transitions
				
				elif act.find('U') != -1:
					action_split = act.split('_')
					action_target = action_split[1]
					action_robot = int(action_split[0][1]) - 1
					success_chance = 1.0 - self._fail_chance
					seq_change_prob = 1 / self._n_valid_locs
					if len(seq_state) < self._seq_length and curr_state[action_robot][0] == 'F' and seq_state.find(action_target[0]) == -1:
						for r_idx in range(self._n_robots):
							if r_idx != action_robot:
								robot_color = curr_state[r_idx][0]
								if robot_color != 'F':
									if action_target[0] == robot_color:
										target = 'F'
									else:
										target = action_target[0]
									if r_idx > action_robot:
										r_state = (curr_state[:action_robot] + ((target,),) + curr_state[action_robot + 1:r_idx] + (('F',),) +
												   curr_state[r_idx + 1:-1])
									else:
										r_state = (curr_state[:r_idx] + (('F',),) + curr_state[r_idx + 1:action_robot] + ((target,),) +
												   curr_state[action_robot + 1:-1])
									if seq_state.find(robot_color) == -1 and len(seq_state) < self._seq_length:
										nxt_seq_state = seq_state + robot_color
									else:
										nxt_seq_state = seq_state
									nxt_state = r_state + ((nxt_seq_state,),)
									nxt_idx = state_lst.index(RobotTeamColorSeqGame.get_state_str(nxt_state))
									p[state_idx][nxt_idx] += seq_change_prob
									success_chance = max(0, success_chance - seq_change_prob)
						
						if seq_state.find(action_target[0]) == -1:
							nxt_robot_state = curr_state[:action_robot] + ((action_target[0],),) + curr_state[action_robot + 1:-1]
						else:
							nxt_robot_state = curr_state[:action_robot] + (('F',),) + curr_state[action_robot + 1:-1]
						nxt_state = nxt_robot_state + ((curr_state[-1][0],),)
						nxt_idx = state_lst.index(RobotTeamColorSeqGame.get_state_str(nxt_state))
						p[state_idx][nxt_idx] += success_chance
						p[state_idx][state_idx] += self._fail_chance
						p[state_idx] = p[state_idx] / p[state_idx].sum()
					
				else:
					print(colored('Action not recognized. Skipping matrix probability', 'red'))
					continue
			
			P[act] = p
		
		return P
	
	@staticmethod
	def generate_rewards(goal_states: List[int], states: np.ndarray, actions: np.ndarray) -> np.ndarray:
		nX = len(states)
		nA = len(actions)
		
		c = np.zeros((nX, nA))
		for state in goal_states:
			c[state, :] = 100.0
		
		return c
	
	@staticmethod
	def generate_costs(goal_states: List[int], states: np.ndarray, actions: np.ndarray) -> np.ndarray:
		nX = len(states)
		nA = len(actions)
		
		c = np.ones((nX, nA))
		for state in goal_states:
			c[state, :] = 0.0
		
		return c
	
	def generate_world(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
		print('### Generating Word Maze World for Team Decision ###')
		print('Generating States')
		states = self.generate_states()
		print('Generating Actions')
		actions = self.generate_actions()
		print('Generating Transitions')
		probabilities = self.generate_stochastic_probabilities(states, actions)
		
		self._states = states
		self._actions = actions
		
		print('World Created')
		return states, actions, probabilities


class RobotTeamColorSeqGame(MDPWorld):
	
	def __init__(self, colors: List[str], start_states: List[Tuple], color_states: List[Tuple], walls: List[Tuple], color_seqs: List[Tuple],
				 n_locs: int, fail_chance: float = 0.0, n_robots: int = 2, seq_length: int = 4):
		self._colors = colors
		self._start_states = start_states
		self._color_states = color_states
		self._color_seqs = color_seqs
		self._walls = walls
		self._fail_chance = fail_chance
		self._n_robots = n_robots
		self._n_valid_locs = n_locs
		self._seq_length = seq_length
		
		self._states = []
		self._actions = []
	
	@staticmethod
	def get_state_str(state_tuple: Tuple) -> str:
		return ', '.join(' '.join(str(x) for x in elem) for elem in state_tuple)
	
	@staticmethod
	def get_state_tuple(state_str: str) -> Tuple:
		state = []
		state_split = state_str.split(', ')
		for elem in state_split:
			elem_split = elem.split(' ')
			elem_tuple = []
			for elem_2 in elem_split:
				try:
					elem_tuple += [int(elem_2)]
				except ValueError:
					elem_tuple += [elem_2]
			state += [tuple(elem_tuple)]
		
		return tuple(state)
	
	@staticmethod
	def get_action_tuple(action_str: str) -> Tuple[str]:
		return tuple(action_str.split(' '))
	
	def generate_states(self) -> np.ndarray:
		robot_states = ['F']
		for letter in self._colors:
			robot_states += [letter]
		states = []
		words = []
		
		for comb in itertools.product(self._colors + ['N'], repeat=self._seq_length):
			comb_str = ''.join(comb)
			add = True
			for i in range(self._seq_length):
				for j in range(i+1, self._seq_length):
					if comb_str[i] != 'N' and comb_str[i] == comb_str[j]:
						add = False
			if comb_str not in words and add:
				words += [comb_str]
		
		for state in itertools.product(robot_states, repeat=self._n_robots):
			for word in words:
				states += [', '.join(state + (''.join(word), ))]
		
		return np.array(states)
	
	def generate_actions(self) -> np.ndarray:
		possible_actions = ['N']
		for letter in self._colors:
			for i in range(self._n_robots):
				for j in range(self._seq_length):
					possible_actions += ['U' + str(i+1) + '_' + letter + str(j+1)]
		return np.array(possible_actions, dtype=tuple)
	
	def action_detailed(self, action: str) -> Tuple:
		
		if action == 'N':
			return 'N', 'self'
		
		elif action.find('U') != -1:
			return 'U', tuple(list(action[-2:])), int(action[1])
		
		else:
			print(colored('Unrecognized action for Team definition of a Word Maze world', 'red'))
			return 'N', 'self'
	
	def generate_stochastic_probabilities(self, states: np.ndarray, actions: np.ndarray) -> Dict[str, np.ndarray]:
		nX = len(states)
		state_lst = list(states)
		
		letter_states = {}
		
		for x, y, l in self._color_states:
			letter_states[l] = (x, y)
		
		P = {}
		
		for act in actions:
			p = np.zeros((nX, nX))
			for state in states:
				curr_state = RobotTeamColorSeqGame.get_state_tuple(state)
				state_idx = state_lst.index(state)
				robot_states = curr_state[:-1]
				seq_state = curr_state[-1][0]
				if act == 'N':
					state_transitions = np.zeros(nX)
					
					if np.all([curr_state[i] == 'F' for i in range(self._n_robots)]):
						state_transitions[state_idx] = 1.0
					
					else:
						success_chance = 1 - self._fail_chance
						seq_change_prob = 1 / self._n_valid_locs
						
						use_robots = [i for i, x in enumerate(robot_states) if x[0] != 'F']
						if len(use_robots) > 0:
							for i in range(len(use_robots), 0, -1):
								use_robots_comb = itertools.combinations(use_robots, i)
								for comb in use_robots_comb:
									nxt_r_state = ()
									nxt_seq_states = []
									trans_prob = seq_change_prob ** max(1, len(comb))
									for idx in range(self._n_robots):
										if idx in comb:
											nxt_r_state += (('F',),)
										else:
											if any([robot_states[idx] == robot_states[c_idx] for c_idx in comb]):
												nxt_r_state += (('F',),)
											else:
												nxt_r_state += (robot_states[idx],)
									for perm in itertools.permutations(comb, len(comb)):
										for p_idx in range(len(perm), 0, -1):
											for active_robots in itertools.permutations(range(self._seq_length), p_idx):
												nxt_seq_state = list(seq_state)
												n_active_robots = len(active_robots)
												if n_active_robots < self._n_robots:
													if n_active_robots < 2:
														if ''.join(nxt_seq_state).find(perm[-1]) == -1:
															nxt_seq_state[active_robots[0]] = perm[-1]
														nxt_seq_states += [nxt_seq_state]
													else:
														for remain_robots in itertools.combinations(range(self._n_robots - 1), n_active_robots - 1):
															non_overlap_robots = list(remain_robots) + [self._n_robots - 1]
															for idx, seq_idx in enumerate(active_robots):
																if ''.join(nxt_seq_state).find(perm[non_overlap_robots[idx]]) == -1:
																	nxt_seq_state[seq_idx] = perm[non_overlap_robots[idx]]
															nxt_seq_states += [nxt_seq_state]
												else:
													for idx, seq_idx in enumerate(active_robots):
														if ''.join(nxt_seq_state).find(perm[idx]) == -1:
															nxt_seq_state[seq_idx] = perm[idx]
													nxt_seq_states += [nxt_seq_state]
									for nxt_seq in nxt_seq_states:
										nxt_state = (nxt_r_state + ((nxt_seq[:self._seq_length],),))
										nxt_idx = state_lst.index(RobotTeamColorSeqGame.get_state_str(nxt_state))
										state_transitions[nxt_idx] += trans_prob
										success_chance -= trans_prob
							
						state_transitions[state_idx] += success_chance
						state_transitions[state_idx] += self._fail_chance
						state_transitions = state_transitions / state_transitions.sum()
						
					p[state_idx] = state_transitions
					
				elif act.find('U') != -1:
					action_split = act.split('_')
					action_target = action_split[1]
					action_robot = int(action_split[0][1]) - 1
					success_chance = 1 - self._fail_chance
					seq_change_prob = 1 / self._n_valid_locs
					if curr_state[action_robot][0] == 'F' and state.find(action_target[0]) == -1:
						for color in self._colors:
							for r_idx in range(self._n_robots):
								for s_idx in range(self._seq_length):
									if r_idx != action_robot:
										if curr_state[r_idx][0] == color:
											if action_target[0] == color:
												target = 'F'
											else:
												target = action_target[0]
											if r_idx > action_robot:
												r_state = (curr_state[:action_robot] + ((target,),) + curr_state[action_robot+1:r_idx] + (('F', ), ) +
														   curr_state[r_idx+1:-1])
											else:
												r_state = (curr_state[:r_idx] + (('F',),) + curr_state[r_idx+1:action_robot] + ((target,),) +
														   curr_state[action_robot+1:-1])
											if curr_state[-1][0].find(color) == -1 and curr_state[-1][0][s_idx] != color:
												seq_state = curr_state[-1][0][:s_idx] + color + curr_state[-1][0][s_idx + 1:]
											else:
												seq_state = curr_state[-1][0]
											nxt_state = r_state + ((seq_state,),)
											nxt_idx = state_lst.index(RobotTeamColorSeqGame.get_state_str(nxt_state))
											p[state_idx][nxt_idx] += seq_change_prob
											success_chance = max(0, success_chance - seq_change_prob)
						
						if curr_state[-1][0].find(action_target[0]) == -1:
							nxt_robot_state = curr_state[:action_robot] + ((action_target[0], ), ) + curr_state[action_robot+1:-1]
						else:
							nxt_robot_state = curr_state[:action_robot] + (('F',),) + curr_state[action_robot + 1:-1]
						nxt_state = nxt_robot_state + ((curr_state[-1][0], ), )
						nxt_idx = state_lst.index(RobotTeamColorSeqGame.get_state_str(nxt_state))
						p[state_idx][nxt_idx] += success_chance
						
						p[state_idx][state_idx] += self._fail_chance
						p[state_idx] = p[state_idx] / p[state_idx].sum()
				
				else:
					print(colored('Action not recognized. Skipping matrix probability', 'red'))
					continue
				
			P[act] = p
		
		return P
	
	@staticmethod
	def generate_rewards(goal_states: List[int], states: np.ndarray, actions: np.ndarray) -> np.ndarray:
		nX = len(states)
		nA = len(actions)
		
		c = np.zeros((nX, nA))
		for state in goal_states:
			c[state, :] = 100.0
		
		return c
	
	@staticmethod
	def generate_costs(goal_states: List[int], states: np.ndarray, actions: np.ndarray) -> np.ndarray:
		nX = len(states)
		nA = len(actions)
		
		c = np.ones((nX, nA))
		for state in goal_states:
			c[state, :] = 0.0
		
		return c
	
	def generate_world(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
		print('### Generating Word Maze World for Team Decision ###')
		print('Generating States')
		states = self.generate_states()
		print('Generating Actions')
		actions = self.generate_actions()
		print('Generating Transitions')
		probabilities = self.generate_stochastic_probabilities(states, actions)
		
		self._states = states
		self._actions = actions
		
		print('World Created')
		return states, actions, probabilities
