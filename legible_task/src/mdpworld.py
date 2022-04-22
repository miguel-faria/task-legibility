#! /usr/bin/env python
from abc import ABC
from typing import List, Tuple, Dict


class MDPWorld(ABC):
	
	def generate_states(self, n_rows, n_cols, obj_states):
		pass
	
	def generate_actions(self):
		pass
	
	def action_detailed(self, action):
		pass
	
	def generate_stochastic_probabilities(self, states, actions, obj_states, max_rows, max_cols, fail_chance):
		pass
	
	def generate_probabilities(self, states, actions, obj_states, max_rows, max_cols):
		pass
	
	def generate_rewards(self, goal, states, actions):
		pass
	
	def generate_costs(self, goal, states, actions):
		pass
	
	def generate_costs_varied(self, goal, states, actions, probabilities):
		pass
	
	def generate_world(self, n_rows: int = 10, n_cols: int = 10, obj_states: List[Tuple[int, int, str]] = None, fail_chance: float = 0.0,
					   prob_type: str = 'stoc', max_grab_objs: int = 10, walls: List = None):
		pass
