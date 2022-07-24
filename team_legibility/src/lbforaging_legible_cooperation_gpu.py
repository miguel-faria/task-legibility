#! /usr/bin/env python
import sys

import numpy as np
import tensorflow as tf
import argparse
import gym
import lbforaging
import pickle
import yaml
import itertools
import time
import os
import subprocess

from pathlib import Path
from gym.envs.registration import register
from lbforaging.foraging.environment import ForagingEnv
from numpy import ndarray

from environments.lb_foraging_plan import ForagingPlan
from termcolor import colored
from typing import List, Tuple, Dict
from agents.tom_agent import ToMAgent
from agents.legible_mdp_agent import LegibleMDPAgent
from utils import policy_iteration_gpu, load_savepoint, store_savepoint
from scipy.sparse import csr_matrix
from tqdm import tqdm
from statistics import stdev
from math import sqrt

# Environment parameters
N_AGENTS = 2
MAX_FOOD = 4
AGENT_LVL = 1
FOOD_LVL = 2
COOP = True
FIELD_LENGTH = 8
AGENT_SIGHT = 8
RNG_SEED = 20220510
MAX_STEPS = 300

ACTION_MAP = {0: 'None', 1: 'Up', 2: 'Down', 3: 'Left', 4: 'Right', 5: 'Load'}
ACTION_MOVE = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1), 5: (0, 0)}


class LeadingAgent:
	"""
		Small class for the interaction's leading agent
	"""
	def __init__(self, q_library: List[np.ndarray], pol_library: List[np.ndarray]):
		
		self._q_library = q_library
		self._pol_library = pol_library
		self._n_states, self._n_actions = self._q_library[0].shape
		self._n_tasks = len(q_library)
		self._task = -1
	
	def opt_acting(self, state: int, rng_gen: np.random.Generator) -> int:
		return rng_gen.choice(self._n_actions, p=self._pol_library[self._task][state, :])
	
	def sub_acting(self, state: int, rng_gen: np.random.Generator, act_try: int) -> int:
		sorted_q = self._q_library[self._task][state].copy()
		sorted_q.sort()
		
		if act_try > len(sorted_q):
			return rng_gen.choice(len(sorted_q))
		
		q_len = len(sorted_q)
		nth_best = sorted_q[max(-(act_try + 1), -q_len)]
		return rng_gen.choice(np.where(self._q_library[self._task][state] == nth_best)[0])
	
	@property
	def task(self) -> int:
		return self._task
	
	@property
	def q_library(self) -> List[np.ndarray]:
		return self._q_library
	
	@property
	def pol_library(self) -> List[np.ndarray]:
		return self._pol_library
	
	@q_library.setter
	def q_library(self, new_library: List[np.ndarray]):
		self._q_library = new_library
	
	@pol_library.setter
	def pol_library(self, new_library: List[np.ndarray]):
		self._pol_library = new_library
	
	def set_task(self, task_id: int):
		self._task = task_id

	def update_decision(self, q_library: List[tf.Tensor], pol_library: List[np.ndarray]):
		self._pol_library = pol_library
		for i in range(len(self._q_library)):
			self._q_library[i] = q_library[i].numpy()


def gpu_query_free_memory() -> Dict:
	"""
	Querry nvidia-smi for Nvidia GPUs free memory
	
	:return gpu_memory_map: dictonary with the free GPU memory indexed to each GPU
	"""
	result = subprocess.check_output([ 'nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], encoding='utf-8')
	gpu_memory = [int(x) for x in result.strip().split('\n')]
	gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
	return gpu_memory_map


def get_gpu_most_free() -> int:
	"""
	Obtain the gpu with more free memory
	
	:return: index for the gpu
	"""
	
	gpu_free_mem = gpu_query_free_memory()
	free_mem = list(gpu_free_mem.values())
	return free_mem.index(max(free_mem))


def convert_csr_to_sparse_tensor(sparse_mat: csr_matrix) -> tf.SparseTensor:
	"""
	Convert a Scipy CSR Sparse matrix to a Tensorflow SparseTensor

	:param sparse_mat: input CSR sparse matrix
	:return: output tensorflow SparseTensor
	"""
	
	coo = sparse_mat.tocoo( )
	indices: ndarray = np.mat([coo.row, coo.col]).transpose( )
	return tf.SparseTensor(indices, coo.data, coo.shape)


def state_valid(nxt_agent_locs: List, agent_locs: List, agent_lvs: List, food_state: int) -> str:
	"""
	Returns the valid next state based on the current and expected next agent locations
	
	:param nxt_agent_locs: list with expected next agent locations
	:param agent_locs: list wit current agent locations
	:param agent_lvs: list with the levels of the different agents
	:param food_state: flag that captures if food has been caught or not
	:return nxt_state: string id of the valid next state
	"""
	
	n_agents = len(agent_locs)
	dups = []
	agent_dups = np.zeros(n_agents)
	for idx in range(n_agents):
		nxt_loc = nxt_agent_locs[idx]
		if nxt_loc not in dups:
			dup = False
			for idx_2 in range(idx + 1, n_agents):
				if nxt_loc == nxt_agent_locs[idx_2]:
					if not dup:
						dup = True
					agent_dups[idx] = 1
					agent_dups[idx_2] = 1
			if dup:
				dups.append(nxt_loc)
	
	nxt_state = ''
	for idx in range(n_agents):
		if agent_dups[idx] > 0:
			nxt_state += ''.join([str(x) for x in agent_locs[idx]]) + str(agent_lvs[idx])
		else:
			nxt_state += ''.join([str(x) for x in nxt_agent_locs[idx]]) + str(agent_lvs[idx])
	nxt_state += str(food_state)
	
	return nxt_state


def update_transitions(states: Tuple, action: str, transitions: csr_matrix, food_locs: List, target_food: Tuple, field_size: Tuple) -> csr_matrix:
	"""
	Action transition updates for the selected target food, given the current field layout and food positions (general single-agent case)
	
	:param states: state space
	:param action: action space
	:param transitions: original sparse transition matrix for current action
	:param food_locs: list with locations of foods in the environment
	:param target_food: location of the current target food
	:param field_size: tuple with the number of rows and columns in the field
	:return new_trans: updated sparse transition matrix for current action
	"""
	
	new_trans = transitions.toarray().copy()
	other_foods = food_locs.copy()
	if target_food in other_foods:
		other_foods.remove(target_food)
	n_rows, n_cols = field_size
	
	for state_idx in range(len(states)):
		
		state = states[state_idx]
		agent_locs, agent_lvls, food_state = ForagingPlan.get_state_tuple(state)
		agent_loc = agent_locs[0]
		
		if action == 'U':
			nxt_pos = (max(agent_loc[0] - 1, 0), agent_loc[1])
		
		elif action == 'D':
			nxt_pos = (min(agent_loc[0] + 1, n_rows - 1), agent_loc[1])
		
		elif action == 'L':
			nxt_pos = (agent_loc[0], max(agent_loc[1] - 1, 0))
		
		elif action == 'R':
			nxt_pos = (agent_loc[0], min(agent_loc[1] + 1, n_cols - 1))
		
		else:
			nxt_pos = agent_loc
		
		if nxt_pos in other_foods:
			nxt_agent_states = np.nonzero(new_trans[state_idx])[0]
			
			for nxt_state_idx in nxt_agent_states:
				
				nxt_state = states[nxt_state_idx]
				nxt_state_true = ''.join([str(x) for x in agent_loc]) + str(agent_lvls[0]) + nxt_state[3:]
				
				if nxt_state_true not in states:
					nxt_state_true = state
				
				new_trans[state_idx, nxt_state_idx] = 0.0
				new_trans[state_idx, states.index(nxt_state_true)] = transitions[state_idx].toarray()[0][nxt_state_idx]
	
	return csr_matrix(new_trans)


def update_agent_transitions(states: Tuple, action: str, food_locs: List, target_food: Tuple, field_size: Tuple,
							 ma_actions: Tuple, ma_pol: np.ndarray) -> tf.SparseTensor:
	"""
	Action transition updates for the selected target food, given the current field layout and food positions
	(single-agent that considers other agents to follow optimal joint policy)
	
	:param states: state space
	:param action: string id for the action
	:param food_locs: list with locations of foods in the environment
	:param target_food: location of the current target food
	:param field_size: tuple with the number of rows and columns in the field
	:param ma_actions: multi-agent joint action space
	:param ma_pol: multi-agent optimal joint action policy
	:return new_trans: updated sparse transition matrix for current action
	"""
	
	nX = len(states)
	new_trans = np.zeros((nX, nX))
	other_foods = food_locs.copy()
	if target_food in other_foods:
		other_foods.remove(target_food)
	n_rows, n_cols = field_size
	
	for state_idx in range(len(states)):
		
		state = states[state_idx]
		opt_pol_act = np.argwhere(ma_pol[state_idx] == ma_pol[state_idx].max()).ravel()
		agent_locs, agent_lvls, food_state = ForagingPlan.get_state_tuple(state)
		
		for pol_act_idx in opt_pol_act:
			
			joint_action = [action] + list(ma_actions[pol_act_idx][1:])
			nxt_agent_locs = []
			agent_idx = 0
			load_lvl_sum = 0
			
			for act in joint_action:
				
				agent_loc = agent_locs[agent_idx]
				
				if act == 'U':
					nxt_pos = (max(agent_loc[0] - 1, 0), agent_loc[1])
				
				elif act == 'D':
					nxt_pos = (min(agent_loc[0] + 1, n_rows - 1), agent_loc[1])
				
				elif act == 'L':
					nxt_pos = (agent_loc[0], max(agent_loc[1] - 1, 0))
				
				elif act == 'R':
					nxt_pos = (agent_loc[0], min(agent_loc[1] + 1, n_cols - 1))
				
				else:
					nxt_pos = agent_loc
					if act == 'Lo':
						agent_adj_locs = adj_locs(agent_loc, field_size)
						if target_food in agent_adj_locs:
							load_lvl_sum += 1
				
				if nxt_pos in other_foods or nxt_pos == target_food:
					nxt_agent_locs += [agent_loc]
				else:
					nxt_agent_locs += [nxt_pos]
				
				agent_idx += 1
			
			nxt_state = state_valid(nxt_agent_locs, agent_locs, agent_lvls, 1 if load_lvl_sum >= FOOD_LVL else food_state)
			new_trans[state_idx, states.index(nxt_state)] += ma_pol[state_idx, pol_act_idx]
	
	return tf.sparse.from_dense(tf.convert_to_tensor(new_trans))


def update_ma_transitions(states: Tuple, joint_action: str, transitions: tf.SparseTensor, food_locs: List, target_food: Tuple, field_size: Tuple) -> tf.SparseTensor:
	"""
	Joint action transition updates for the selected target food, given the current field layout and food positions (general multi-agent case)
	
	:param states: state space
	:param joint_action: string id for the current joint action
	:param transitions: original sparse transition matrix for current joint action
	:param food_locs: list with locations of foods in the environment
	:param target_food: location of the current target food
	:param field_size: tuple with the number of rows and columns in the field
	:return new_trans: updated sparse transition matrix for current joint action
	"""
	
	new_trans = tf.sparse.to_dense(transitions).numpy()
	other_foods = food_locs.copy()
	if target_food in other_foods:
		other_foods.remove(target_food)
	n_rows, n_cols = field_size
	single_acts = list(joint_action)
	
	for state_idx in range(len(states)):
		
		state = states[state_idx]
		agent_locs, agent_lvls, food_state = ForagingPlan.get_state_tuple(state)
		agent_idx = 0
		nxt_agent_locs = []
		nxt_state_idx = np.nonzero(new_trans[state_idx])[0]
		need_update = False
		
		for action in single_acts:
			
			agent_loc = agent_locs[agent_idx]
			
			if action == 'U':
				nxt_pos = (max(agent_loc[0] - 1, 0), agent_loc[1])
			
			elif action == 'D':
				nxt_pos = (min(agent_loc[0] + 1, n_rows - 1), agent_loc[1])
			
			elif action == 'L':
				nxt_pos = (agent_loc[0], max(agent_loc[1] - 1, 0))
			
			elif action == 'R':
				nxt_pos = (agent_loc[0], min(agent_loc[1] + 1, n_cols - 1))
			
			else:
				nxt_pos = agent_loc
			
			if nxt_pos in other_foods:
				need_update = True
				nxt_agent_locs += [agent_loc]
			elif nxt_pos == target_food:
				nxt_agent_locs += [agent_loc]
			else:
				nxt_agent_locs += [nxt_pos]
			
			agent_idx += 1
		
		if need_update:
			nxt_state = state_valid(nxt_agent_locs, agent_locs, agent_lvls, food_state)
			new_trans[state_idx, nxt_state_idx] = 0.0
			new_trans[state_idx, states.index(nxt_state)] = 1.0
	
	return tf.sparse.from_dense(tf.convert_to_tensor(new_trans))


def update_decision(states: Tuple, actions: Tuple, transitions: Dict[str, List[csr_matrix]], spawn_foods: List, food_locs: List, field_size: Tuple, mode: int,
					rewards: List, opt_pols_library: List, opt_q_library: List, leg_pols_library: List, leg_q_library: List, ma_actions: Tuple,
					ma_transitions: Dict[str, List[csr_matrix]], ma_rewards: Dict[str, np.ndarray], ma_pols_library: List, ma_q_library: List) -> Tuple:
	"""
	Update the leader and follower's legible and optimal decision models with current field layout and food locations
	
	:param states: single-agent state space
	:param actions: single-agent action space
	:param transitions: single agent transition matrices
	:param spawn_foods: list of foods in the field
	:param food_locs: list with the locations of all foods spawned in the field
	:param field_size: number of rows and columns in the field
	:param mode: integer code for the agent team composition
	:param rewards: list with the rewards for the follower and leader
	:param opt_pols_library: library with the optimal policies for both the leader and follower
	:param opt_q_library: library with the optimal q-values for both the leader and follower
	:param leg_pols_library: library with the legible policies for both the leader and follower
	:param leg_q_library: library with the legible q-values for both the leader and follower
	:param ma_actions: multi-agent joint action space
	:param ma_transitions: multi-agent joint action transition matrices
	:param ma_rewards: multi-agent joint action reward functions
	:param ma_pols_library: multi-agent optimal joint policies
	:param ma_q_library: multi-agent optimal q-values
	:return: tuple with the updated optimal and legible decision models for the leader and follower agents
	"""
	
	updated_transitions = transitions.copy()
	leader_rewards = rewards[0]
	follower_rewards = rewards[1]
	l_opt_pol_library = opt_pols_library[0].copy()
	l_opt_q_library = opt_q_library[0].copy()
	l_leg_pol_library = leg_pols_library[0].copy()
	l_leg_q_library = leg_q_library[0].copy()
	f_opt_pol_library = opt_pols_library[1].copy()
	f_opt_q_library = opt_q_library[1].copy()
	f_leg_pol_library = leg_pols_library[1].copy()
	f_leg_q_library = leg_q_library[1].copy()
	
	for food in tqdm(spawn_foods, desc='Update dynamics and policies with current field'):
		task_idx = food_locs.index(food)
		task_key = ''.join([str(x) for x in food])
		updated_ma_transitions = []
		for joint_act_idx in range(len(ma_actions)):
			updated_ma_transitions += [update_ma_transitions(states, ma_actions[joint_act_idx], ma_transitions[task_key][joint_act_idx],
															 spawn_foods, food, field_size)]
		ma_opt_pol, _ = policy_iteration_gpu((states, ma_actions, updated_ma_transitions, ma_rewards[task_key], 0.9),
											 ma_pols_library[task_idx], ma_q_library[task_idx])
		update_food_transitions = []
		for act_idx in range(len(actions)):
			update_food_transitions += [update_agent_transitions(states, actions[act_idx], spawn_foods, food, field_size, ma_actions, ma_opt_pol)]
		updated_transitions[task_key] = update_food_transitions
		
		leader_opt_pol, leader_opt_q = policy_iteration_gpu((states, actions, updated_transitions[task_key], leader_rewards[task_key], 0.9),
															l_opt_pol_library[task_idx], l_opt_q_library[task_idx])
		follower_opt_pol, follower_opt_q = policy_iteration_gpu((states, actions, updated_transitions[task_key], follower_rewards[task_key], 0.9),
																f_opt_pol_library[task_idx], f_opt_q_library[task_idx])
		l_opt_pol_library[task_idx] = leader_opt_pol
		l_opt_q_library[task_idx] = leader_opt_q
		f_opt_pol_library[task_idx] = follower_opt_pol
		f_opt_q_library[task_idx] = follower_opt_q
	
	if mode == 1 or mode == 3:
		leader_q_library = []
		for idx in range(len(l_opt_q_library)):
			leader_q_library += [l_opt_q_library[idx].numpy()]
		leader_legible_mdp = LegibleMDPAgent(states, actions, list(updated_transitions.values()), 0.9, False, list(transitions.keys()),
											 0.75, 1, leader_q_library)
		for food_idx in tqdm(range(len(food_locs)), desc='Update leader legible policy with current field'):
			task_key = ''.join([str(x) for x in food_locs[food_idx]])
			leg_pol, leg_q = policy_iteration_gpu((states, actions, updated_transitions[task_key], leader_legible_mdp.costs[food_idx], 0.9),
												  l_leg_pol_library[food_idx], l_leg_q_library[food_idx])
			l_leg_pol_library[food_idx] = leg_pol
			l_leg_q_library[food_idx] = leg_q
	if mode == 2 or mode == 3:
		follower_q_library = []
		for idx in range(len(f_opt_q_library)):
			follower_q_library += [f_opt_q_library[idx].numpy()]
		follower_legible_mdp = LegibleMDPAgent(states, actions, list(updated_transitions.values()), 0.9, False, list(transitions.keys()),
											   0.75, 1, follower_q_library)
		for food_idx in tqdm(range(len(food_locs)), desc='Update follower legible policy with current field'):
			task_key = ''.join([str(x) for x in food_locs[food_idx]])
			leg_pol, leg_q = policy_iteration_gpu((states, actions, updated_transitions[task_key], follower_legible_mdp.costs[food_idx], 0.9),
												  f_leg_pol_library[food_idx], f_leg_q_library[food_idx])
			f_leg_pol_library[food_idx] = leg_pol
			f_leg_q_library[food_idx] = leg_q
	
	return (l_opt_pol_library, l_opt_q_library, l_leg_pol_library, l_leg_q_library,
			f_opt_pol_library, f_opt_q_library, f_leg_pol_library, f_leg_q_library)


def get_state(observation: np.ndarray, food_picked: int, n_agents: int, max_food: int) -> str:
	"""
	Obtain the current agent state from the agent's observation

	:param observation: array with the observed fruit and agents' positions
	:param food_picked: integer flag that captures if the food has been picked
	:param n_agents: number of agents in the field
	:param max_food: maximum number of food items in the field
	:return state: string id of the current agent's state
	"""
	
	state = ''
	
	for i in range(n_agents):
		agent_offset = 3 * max_food + 3 * i
		agent_loc = observation[agent_offset:agent_offset + 2]
		agent_lvl = observation[agent_offset + 2]
		state += ''.join([str(int(x)) for x in agent_loc]) + str(int(agent_lvl))
	
	state += str(food_picked)
	
	return state


def adj_locs(loc: Tuple, field_size: Tuple) -> List[Tuple]:
	"""
	Obtain the list of locations adjacent to a specified location
	
	:param loc: target location to obtain adjacent locations
	:param field_size: number of rows and columns in the field
	:return: list of valid adjacent locations
	"""
	
	return [(min(loc[0] + 1, field_size[0]), loc[1]), (max(loc[0] - 1, 0), loc[1]), (loc[0], min(loc[1] + 1, field_size[1])), (loc[0], max(loc[1] - 1, 0))]


def spawn_food(field: np.ndarray, food_locs: List[Tuple], rng_gen: np.random.Generator, field_size: Tuple, max_food: int) -> np.ndarray:
	"""
	Randomly spawn foods in the current field, according to supplied food locations

	:param field: current field layout
	:param food_locs: list of possible food locations
	:param rng_gen: random number generator
	:param field_size: field number of rows and columns
	:param max_food: maximum food items in the field
	:return new_field: new field layout with foods spawned
	"""
	
	new_field = field.copy()
	foods = list([tuple(x) for x in np.argwhere(field > 0)])
	n_foods = np.count_nonzero(new_field)
	n_food_locs = len(food_locs)
	
	if n_foods < max_food:
		done = False
		while not done:
			food_idx = rng_gen.choice(n_food_locs)
			food_loc = food_locs[food_idx]
			food_loc_adj = adj_locs(food_loc, field_size)
			if food_loc not in foods and not any([True if loc in foods else False for loc in food_loc_adj]):
				new_field[food_locs[food_idx]] = FOOD_LVL
				done = True
	
	return new_field


def load_env_model(filename) -> Tuple[Tuple, Tuple, Dict, Dict]:
	"""
	Load environment model from file
	
	:param filename: filename with stored environment
	:return: loaded environment model (state space, action space, transition matrices, rewards library)
	"""
	
	model_dir = Path(__file__).parent.absolute().parent.absolute() / 'models'
	model_file = model_dir / (filename + '.pkl')
	data = pickle.load(open(model_file, 'rb'))
	return data['states'], data['actions'], data['transitions'], data['rewards']


def load_decision_model(filename) -> Tuple[List[np.ndarray], List[np.ndarray]]:
	"""
	Load decision model from file
	
	:param filename: filename with stored decision model
	:return: loaded decision model (q-matrix and policy)
	"""
	
	model_dir = Path(__file__).parent.absolute().parent.absolute() / 'models'
	model_file = model_dir / (filename + '.pkl')
	data = pickle.load(open(model_file, 'rb'))
	return data['q_matrix'], data['pol']


def is_deadlock(history: List, new_states: Tuple, n_agents: int) -> bool:
	"""
	Detector of repeated states in the agents' recent history
	
	:param history: list with the agents' states and actions
	:param new_states: list with the next agents' states
	:param n_agents: number of agents in the field
	:return deadlock: true if agents' are in a deadlock
	"""
	
	if len(history) < 3:
		return False
	
	last_states = ()
	last_actions = ()
	for step in history[-3:]:
		for agent_idx in range(n_agents):
			last_states += (step[n_agents * agent_idx],)
	for agent_idx in range(n_agents):
		last_actions += (history[-1][n_agents * agent_idx + 1], )
	
	deadlock = True
	if all([act == 'N' for act in last_actions]) or all([act == 'Lo' for act in last_actions]):
		deadlock = False
	else:
		state_repetition = 0
		for state in new_states:
			for hist_state in last_states:
				if state in hist_state:
					state_repetition += 1
		if state_repetition < 3 * n_agents:
			deadlock = False
	
	return deadlock


def agent_coordination(leader_loc: Tuple, follower_loc: Tuple, actions: Tuple[int, int], field_size: Tuple) -> Tuple[int, int]:
	"""
	Joint action coordination to prevent agents blocking each other, following the social convention that the follower yields to the leader
	
	:param leader_loc: leader current location
	:param follower_loc: follower current location
	:param actions: non-coordinated agent actions
	:param field_size: field's number of rows and columns
	:return new_actions: agents' coordinated joint action
	"""
	
	rows, cols = field_size
	leader_move = ACTION_MOVE[actions[0]]
	follower_move = ACTION_MOVE[actions[1]]
	nxt_leader_loc = (max(min(leader_loc[0] + leader_move[0], rows), 0), max(min(leader_loc[1] + leader_move[1], cols), 0))
	nxt_follower_loc = (max(min(follower_loc[0] + follower_move[0], rows), 0), max(min(follower_loc[1] + follower_move[1], cols), 0))
	
	if nxt_leader_loc == nxt_follower_loc:
		return actions[0], list(ACTION_MAP.keys())[list(ACTION_MAP.values()).index('None')]
	else:
		return actions


def failed_pickup(observation: np.ndarray, last_actions: Tuple, food_pos: Tuple, field_size: Tuple, max_foods: int) -> bool:
	"""
	Verifies if agents tried to correctly pick the food item, but environment did not recognize correct pickup

	:param observation: agents' observation after last step actions
	:param last_actions: most recent actions executed
	:param food_pos: position of the food item to be picked
	:param field_size: tuple with the field's number of cols and rows
	:param max_foods: maximum number of foods in environment
	:return: flag marking if there was a pickup failure error
	"""
	
	food_adj_locs = adj_locs(food_pos, field_size)
	n_agents = len(last_actions)
	agent_locs = [tuple(observation[i][3 * max_foods:3 * max_foods + 2]) for i in range(n_agents)]
	loading_agents = []
	for i in range(n_agents):
		if ACTION_MAP[last_actions[i]] == 'Load' and agent_locs[i] in food_adj_locs:
			loading_agents += [i]
	loading_agents_lvl = int(len(loading_agents)) * AGENT_LVL
	
	return loading_agents_lvl >= FOOD_LVL


def eval_behaviour(nruns: int, nagents: int, max_food: int, env: ForagingEnv, mode: int, sa_model: Tuple, leader_decision: Tuple, follower_decision: Tuple,
				   ma_model: Tuple, ma_decision: Tuple, fields: List, food_locs: List, use_render: bool, data_dir: Path,
				   log_dir: Path, filename_prefix: str, verbose: bool) -> Tuple:
	"""
	Evaluation of the cooperation in a level-based foraging scenario giving a team composition
	
	:param nruns: number of runs for the evaluation cycle
	:param nagents: number of agents in the field
	:param max_food: maximum number of food items in the field
	:param env: level-based foraging environment
	:param mode: mode used for the team compostion
	:param sa_model: single-agent environment model for leader and follower
	:param leader_decision: leader agent's optimal and legible decision model
	:param follower_decision: follower agent's optimal and legible decision model
	:param ma_model: multi-agent environment model
	:param ma_decision: multi-agent optimal joint decision model
	:param fields: list of field layouts to evaluate the cooperation (used for result comparison across different team compositions)
	:param food_locs: list of possible food locations
	:param use_render: flag that controls if evaluation runs are rendered for visualization
	:param data_dir: path for the data folder
	:param log_dir: path for the logging folder
	:param filename_prefix: logging filename prefix
	:param verbose: flag that controls if logging execution is minimal or verbose
	:return: tuple with average steps to capture all food items, average number of steps for follower to correctly predict the current food item, list with the number of steps
	per evaluation run and list with number of steps for correct prediction per evaluation run
	"""
	
	if verbose:
		tf.debugging.set_log_device_placement(True)
	
	# Verify if a savepoint exists to restart from
	savepoint_file = data_dir / 'results' / (filename_prefix + '_' + str(mode) + '.save')
	if savepoint_file.exists() and os.path.getsize(savepoint_file) > 0:
		print('Restarting evaluation. Loading savepoint.')
		eval_results, eval_begin = load_savepoint(savepoint_file)
		avg_run_steps = eval_results['avg_steps']
		avg_run_predict_steps = eval_results['avg_predict']
		run_steps = eval_results['run_steps']
		run_predict_steps = eval_results['predict_steps']
		n_errors = eval_results['n_errors']
		error_runs = eval_results['error_runs']
		eval_history = eval_results['history']
		
		# Setting logging outputs in append mode to continue evaluation log
		sys.stdout = open(log_dir / (filename_prefix + '_' + str(mode) + '_log.txt'), 'a')
		sys.stderr = open(log_dir / (filename_prefix + '_' + str(mode) + '_err.txt'), 'a')
	# If no savepoint exists starts evaluation anew
	else:
		avg_run_steps = 0
		avg_run_predict_steps = 0
		run_steps = []
		run_predict_steps = []
		eval_history = []
		eval_begin = 0
		n_errors = 0
		error_runs = []
		
		# Setting logging outputs in write mode to start new evaluation log
		sys.stdout = open(log_dir / (filename_prefix + '_' + str(mode) + '_log.txt'), 'w')
		sys.stderr = open(log_dir / (filename_prefix + '_' + str(mode) + '_err.txt'), 'w')
	
	# Eval parameters setup
	print('Running eval for team composition %d' % mode)
	env.seed(RNG_SEED)
	plan_states, plan_actions, transitions, leader_rewards, follower_rewards = sa_model
	leader_opt_q_library, leader_opt_pol_library, leader_leg_q_library, leader_leg_pol_library = leader_decision
	follower_opt_q_library, follower_opt_pol_library, follower_leg_q_library, follower_leg_pol_library = follower_decision
	plan_actions_ma, transitions_ma, rewards_ma = ma_model
	optimal_q_ma, optimal_pol_ma = ma_decision
	
	# Agent team setup
	if mode == 0:
		leader_q_library = []
		follower_q_library = []
		for idx in range(len(leader_opt_q_library)):
			leader_q_library += [leader_opt_q_library[idx].numpy()]
			follower_q_library += [follower_opt_q_library[idx].numpy()]
		leading_agent = LeadingAgent(leader_q_library, leader_opt_pol_library)
		follower_agent = ToMAgent(follower_q_library, 1, leader_q_library)
	elif mode == 1:
		leader_q_library = []
		follower_q_library = []
		for idx in range(len(leader_opt_q_library)):
			leader_q_library += [leader_leg_q_library[idx].numpy()]
			follower_q_library += [follower_opt_q_library[idx].numpy()]
		leading_agent = LeadingAgent(leader_q_library, leader_leg_pol_library)
		follower_agent = ToMAgent(follower_q_library, 1, leader_q_library)
	elif mode == 2:
		leader_q_library = []
		follower_q_library = []
		for idx in range(len(leader_opt_q_library)):
			leader_q_library += [leader_leg_q_library[idx].numpy()]
			follower_q_library += [follower_leg_q_library[idx].numpy()]
		leading_agent = LeadingAgent(leader_q_library, leader_leg_pol_library)
		follower_agent = ToMAgent(follower_q_library, 1, leader_q_library)
	elif mode == 3:
		leader_q_library = []
		follower_q_library = []
		for idx in range(len(leader_opt_q_library)):
			leader_q_library += [leader_opt_q_library[idx].numpy()]
			follower_q_library += [follower_leg_q_library[idx].numpy()]
		leading_agent = LeadingAgent(leader_q_library, leader_opt_pol_library)
		follower_agent = ToMAgent(follower_q_library, 1, leader_q_library)
	else:
		print(colored('[Error] Invalid execution mode: %d. Stopping execution' % mode), 'red')
		return -1, -1, [], [], []
	
	deadlock_states = []
	rng_gen = np.random.default_rng(RNG_SEED)
	
	# Evaluation cycle
	for run_n in range(eval_begin, nruns):
		print('Starting run %d' % (run_n + 1))
		print('Environment setup')
		env.reset()
		field = fields[min(run_n, len(fields) - 1)]
		print('Field: ')
		print(field)
		env.set_field(field.copy())
		env.spawn_players(AGENT_LVL + 1)
		observation, _, _, _ = env.step((plan_actions.index('N'), plan_actions.index('N')))
		n_spawn_foods = np.count_nonzero(env.field)
		spawn_foods = [tuple(x) for x in np.transpose(np.nonzero(field))]
		follower_agent.set_task_list([food_locs.index(food) for food in spawn_foods])
		print(follower_agent.task_list)
		new_decision_model = update_decision(plan_states, plan_actions, transitions, spawn_foods, food_locs, env.field_size, mode,
											 [leader_rewards, follower_rewards], [leader_opt_pol_library, follower_opt_pol_library],
											 [leader_opt_q_library, follower_opt_q_library], [leader_leg_pol_library, follower_leg_pol_library],
											 [leader_leg_q_library, follower_leg_q_library], plan_actions_ma, transitions_ma, rewards_ma,
											 optimal_pol_ma, optimal_q_ma)
		leader_decision_model = new_decision_model[:4]
		follower_decision_model = new_decision_model[4:]
		if mode == 0:
			leading_agent.update_decision(leader_decision_model[1], leader_decision_model[0])
			follower_agent.update_decision_gpu(follower_decision_model[1], leader_decision_model[1])
		elif mode == 1:
			leading_agent.update_decision(leader_decision_model[3], leader_decision_model[2])
			follower_agent.update_decision_gpu(follower_decision_model[1], leader_decision_model[3])
		elif mode == 2:
			leading_agent.update_decision(leader_decision_model[3], leader_decision_model[2])
			follower_agent.update_decision_gpu(follower_decision_model[3], leader_decision_model[3])
		elif mode == 3:
			leading_agent.update_decision(leader_decision_model[1], leader_decision_model[0])
			follower_agent.update_decision_gpu(follower_decision_model[3], leader_decision_model[1])
		
		leader_state = get_state(observation[0], 0, nagents, max_food)
		follower_state = get_state(observation[1], 0, nagents, max_food)
		leader_state = plan_states.index(leader_state)
		follower_state = plan_states.index(follower_state)
		opt_q_lib = leader_decision_model[1]
		valid_food = False
		while not valid_food:
			food_idx = food_locs.index(tuple(rng_gen.choice(spawn_foods)))
			if opt_q_lib[food_idx][leader_state].sum() > 0.0:
				leading_agent.set_task(food_idx)
				valid_food = True
		
		actions = (leading_agent.opt_acting(leader_state, rng_gen),
				   follower_agent.action(follower_state, (leader_state, 0), 1.0, rng_gen))
		
		done = False
		pick_error = False
		n_steps = 0
		run_history = []
		history = [[food_locs[leading_agent.task], plan_states[leader_state], ACTION_MAP[actions[0]], plan_states[follower_state], ACTION_MAP[actions[1]]]]
		n_pred_steps = []
		act_try = 0
		later_error = 0
		later_food_step = 0
		pred_step = 0
		
		if use_render:
			env.render()
		
		print('Environment setup complete. Starting evaluation')
		while not done:
			n_steps += 1
			
			if use_render:
				env.render()
			last_leader_sample = (leader_state, actions[0])
			if leading_agent.task != follower_agent.task_inference():
				later_error = n_steps
			observation, _, _, _ = env.step(actions)
			current_food_count = np.count_nonzero(env.field)
			
			if current_food_count < 1 or n_steps > MAX_STEPS:
				done = True
				run_history += [history]
				history = []
				if n_steps > MAX_STEPS:
					print('Couldn\'t pick all foods in time')
			
			elif current_food_count < n_spawn_foods:
				print('Food caught')
				n_spawn_foods = current_food_count
				leader_state = plan_states.index(get_state(observation[0], 1, nagents, max_food))
				follower_state = plan_states.index(get_state(observation[1], 1, nagents, max_food))
				food_left = [tuple(x) for x in np.transpose(np.nonzero(env.field))]
				run_history += [history]
				n_pred_steps += [(later_error - later_food_step)]
				pred_step += (later_error - later_food_step) / (max_food - 1)
				later_food_step = n_steps
				later_error = n_steps
				history = []
				
				new_decision_model = update_decision(plan_states, plan_actions, transitions, food_left, food_locs, env.field_size, mode,
													 [leader_rewards, follower_rewards], [leader_opt_pol_library, follower_opt_pol_library],
													 [leader_opt_q_library, follower_opt_q_library], [leader_leg_pol_library, follower_leg_pol_library],
													 [leader_leg_q_library, follower_leg_q_library], plan_actions_ma, transitions_ma, rewards_ma,
													 optimal_pol_ma, optimal_q_ma)
				leader_decision_model = new_decision_model[:4]
				follower_decision_model = new_decision_model[4:]
				if mode == 0:
					leading_agent.update_decision(leader_decision_model[1], leader_decision_model[0])
					follower_agent.update_decision_gpu(follower_decision_model[1], leader_decision_model[1])
				elif mode == 1:
					leading_agent.update_decision(leader_decision_model[3], leader_decision_model[2])
					follower_agent.update_decision_gpu(follower_decision_model[1], leader_decision_model[3])
				elif mode == 2:
					leading_agent.update_decision(leader_decision_model[3], leader_decision_model[2])
					follower_agent.update_decision_gpu(follower_decision_model[3], leader_decision_model[3])
				elif mode == 3:
					leading_agent.update_decision(leader_decision_model[1], leader_decision_model[0])
					follower_agent.update_decision_gpu(follower_decision_model[3], leader_decision_model[1])
				
				opt_q_lib = leader_decision_model[1]
				valid_food = False
				while not valid_food:
					nxt_food_idx = food_locs.index(tuple(rng_gen.choice(food_left)))
					if opt_q_lib[nxt_food_idx][leader_state].sum() > 0.0:
						leading_agent.set_task(nxt_food_idx)
						valid_food = True
				follower_agent.reset_inference([food_locs.index(food) for food in food_left])
				last_leader_sample = (leader_state, 0)
			
			else:
				if failed_pickup(observation, actions, food_locs[leading_agent.task], env.field_size, max_food):
					print('######################################################\n'
						  '## ENVIRONMENT PICKUP ERROR!!!!!! Ignoring eval run ##\n'
						  '######################################################\n')
					pick_error = True
					break
				leader_state = plan_states.index(get_state(observation[0], 0, nagents, max_food))
				follower_state = plan_states.index(get_state(observation[1], 0, nagents, max_food))
			
			if is_deadlock(history, (plan_states[leader_state], plan_states[follower_state]), nagents):
				if not deadlock_states:
					deadlock_states += [(leading_agent.task, plan_states[leader_state])]
					deadlock_states += [(follower_agent.task_inference(), plan_states[follower_state])]
				else:
					if not any(plan_states[leader_state] == s_state[1] for s_state in deadlock_states):
						deadlock_states += [(leading_agent.task, plan_states[leader_state])]
					if not any(plan_states[follower_state] == s_state[1] for s_state in deadlock_states):
						deadlock_states += [(follower_agent.task_inference(), plan_states[follower_state])]
				act_try += 1
				actions = (leading_agent.sub_acting(leader_state, rng_gen, act_try),
						   follower_agent.sub_acting(follower_state, rng_gen, act_try, last_leader_sample, 1.0))
			else:
				act_try = 0
				actions = (leading_agent.opt_acting(leader_state, rng_gen),
						   follower_agent.action(follower_state, last_leader_sample, 1.0, rng_gen))
			
			leader_loc = observation[0][3 * max_food + 0:3 * max_food + 2]
			follower_loc = observation[1][3 * max_food + 0:3 * max_food + 2]
			actions = agent_coordination(leader_loc, follower_loc, actions, env.field_size)
			history += [[food_locs[leading_agent.task], plan_states[leader_state], ACTION_MAP[actions[0]], plan_states[follower_state], ACTION_MAP[actions[1]]]]
			if use_render:
				time.sleep(0.15)
				# input()
		
		if use_render:
			env.render()
		
		follower_agent.reset_inference()
		print('Run Over!!')
		if not pick_error:
			run_steps += [n_steps]
			run_predict_steps += [n_pred_steps]
			avg_run_steps += n_steps / nruns
			avg_run_predict_steps += pred_step / nruns
			eval_history += [run_history]
		else:
			n_errors += 1
			error_runs += [run_n]
		
		curr_results = {'avg_steps': avg_run_steps, 'run_steps': run_steps, 'avg_predict': avg_run_predict_steps,
						'predict_steps': run_predict_steps, 'n_errors': n_errors, 'error_runs': error_runs, 'history': eval_history}
		store_savepoint(savepoint_file, curr_results, run_n)
	
	# Print evaluation results
	print('Number of Deadlocks %d' % int(len(deadlock_states)))
	print('Number of failures due to error: %d' % n_errors)
	print('Error runs: ' + str(error_runs))
	if len(run_steps) > 1:
		print('Average number of steps: %.2f and std error of %.2f' % (avg_run_steps, stdev(run_steps) / sqrt(nruns)))
	else:
		print('Average number of steps: %.2f and std error of %.2f' % (avg_run_steps, 0))
	avg_run_preds = []
	for run_preds in run_predict_steps:
		n_preds = len(run_preds)
		if n_preds > 0:
			run_avg = 0
			for pred in run_preds:
				run_avg += pred / n_preds
			avg_run_preds += [run_avg]
	if len(run_steps) > 1:
		print('Average steps to correct guess: %.2f and std error of %.2f' % (avg_run_predict_steps, stdev(avg_run_preds) / sqrt(nruns)))
	else:
		print('Average steps to correct guess: %.2f and std error of %.2f' % (avg_run_predict_steps, 0))
	print(run_steps)
	print(run_predict_steps)
	
	return avg_run_steps, avg_run_predict_steps, run_steps, run_predict_steps, eval_history


def write_full_results_csv(csv_file: Path, results: Dict, access_type: str, fields: List[str]) -> None:
	"""
	Results writing to csv file
	
	:param csv_file: path to output csv file
	:param results: dictionary with results ordered by team composition code
	:param access_type: string code for type of access to csv file
	:param fields: field headers for csv file
	:return: None
	"""
	
	try:
		with open(csv_file, access_type) as csvfile:
			rows = []
			for key in sorted(results.keys()):
				row = [key]
				inner_keys = list(results[key].keys())
				for inner_key in inner_keys:
					row += [results[key][inner_key]]
				rows += [row]
			
			headers = ', '.join(fields)
			np.savetxt(fname=csvfile, X=np.array(rows, dtype=object), delimiter=',', header=headers, fmt='%s', comments='')
		
		print('Results written to file')
	
	except IOError as e:
		print(colored("I/O error: " + str(e), color='red'))


def main():
	
	parser = argparse.ArgumentParser(description='LB-Foraging cooperation scenario using legible TOM with 2 agents and 2 food items')
	parser.add_argument('--mode', dest='mode', type=int, required=True, choices=[0, 1, 2, 3], nargs='+',
						help='List with team composition modes:'
							 '\n\t0 - Optimal agent controls interaction with an optimal follower '
							 '\n\t1 - Legible agent controls interaction with a legible follower '
							 '\n\t2 - Legible agent controls interaction with an optimal follower'
							 '\n\t3 - Optimal agent controls interaction with a legible follower')
	parser.add_argument('--runs', dest='nruns', type=int, required=True, help='Number of trial runs to obtain eval')
	parser.add_argument('--render', dest='render', action='store_true', help='Activate the render to see the interaction')
	parser.add_argument('--verbose', dest='verbose', action='store_true', help='Add verbose logging')
	parser.add_argument('--nagents', dest='agents', type=int, default=N_AGENTS,
						help='Number of agents in the field')
	parser.add_argument('--nfood', dest='foods', type=int, default=MAX_FOOD,
						help='Number of food items in the field')
	parser.add_argument('--field_length', dest='field_length', type=int, default=FIELD_LENGTH,
						help='Length of the square field for the interaction')
	
	args = parser.parse_args()
	team_comps = args.mode
	n_runs = args.nruns
	use_render = args.render
	verbose = args.verbose
	field_length = args.field_length
	
	# Setup GPU
	gpu_more_mem = get_gpu_most_free()
	logical_gpu = tf.config.list_logical_devices()
	if verbose:
		print('GPU: %d' % gpu_more_mem)
		print(logical_gpu[gpu_more_mem + 1].name)
	
	register(
			id="Foraging-{0}x{0}-{1}p-{2}f{3}-v1".format(field_length, args.agents, args.foods, "-coop" if COOP else ""),
			entry_point="lbforaging.foraging:ForagingEnv",
			kwargs={
					"players":           args.agents,
					"max_player_level":  AGENT_LVL + 1,
					"field_size":        (field_length, field_length),
					"max_food":          args.foods,
					"sight":             field_length,
					"max_episode_steps": 500000,
					"force_coop":        COOP,
			},
	)
	
	env = gym.make("Foraging-{0}x{0}-{1}p-{2}f{3}-v1".format(field_length, args.agents, args.foods, "-coop" if COOP else ""))
	filename_prefix = 'lbforaging_' + str(field_length) + 'x' + str(field_length) + '_a' + str(args.agents) + 'l' + str(FOOD_LVL)
	
	with tf.device(logical_gpu[gpu_more_mem + 1].name):
		if verbose:
			print('Loading SA Env')
		states, actions_sa, transitions_sa, leader_rewards = load_env_model(filename_prefix + '_leader_environment')
		_, _, _, follower_rewards = load_env_model(filename_prefix + '_follower_environment')
		for key in transitions_sa.keys():
			np_transitions = transitions_sa[key]
			tensor_transitions = []
			for transition in np_transitions:
				tensor_transitions += [convert_csr_to_sparse_tensor(transition)]
			transitions_sa[key] = tensor_transitions
			leader_rewards[key] = tf.convert_to_tensor(leader_rewards[key])
			follower_rewards[key] = tf.convert_to_tensor(follower_rewards[key])
		
		if verbose:
			print('Loading SA Decision')
		leader_opt_decision = load_decision_model(filename_prefix + '_leader_optimal_decision')
		leader_leg_decision = load_decision_model(filename_prefix + '_leader_legible_decision')
		follower_opt_decision = load_decision_model(filename_prefix + '_follower_optimal_decision')
		follower_leg_decision = load_decision_model(filename_prefix + '_follower_legible_decision')
		for i in range(len(leader_opt_decision[0])):
			leader_opt_decision[0][i] = tf.convert_to_tensor(leader_opt_decision[0][i])
			leader_leg_decision[0][i] = tf.convert_to_tensor(leader_leg_decision[0][i])
			follower_opt_decision[0][i] = tf.convert_to_tensor(follower_opt_decision[0][i])
			follower_leg_decision[0][i] = tf.convert_to_tensor(follower_leg_decision[0][i])
		
		if verbose:
			print('Load MA Env and Decision')
		_, actions_ma, transitions_ma, rewards_ma = load_env_model(filename_prefix + '_ma_environment')
		optimal_ma_decision = load_decision_model(filename_prefix + '_ma_optimal_decision')
		for key in transitions_ma.keys():
			np_transitions = transitions_ma[key]
			tensor_transitions = []
			for transition in np_transitions:
				tensor_transitions += [convert_csr_to_sparse_tensor(transition)]
			transitions_ma[key] = tensor_transitions
			rewards_ma[key] = tf.convert_to_tensor(rewards_ma[key])
		for i in range(len(optimal_ma_decision[0])):
			optimal_ma_decision[0][i] = tf.convert_to_tensor(optimal_ma_decision[0][i])
		
		if verbose:
			print('Load Foraging parameters')
		data_dir = Path(__file__).parent.absolute().parent.absolute() / 'data'
		with open(data_dir / 'configs' / 'lbforaging_plan_configs.yaml') as file:
			config_params = yaml.full_load(file)
			dict_idx = str(field_length) + 'x' + str(field_length) + '_food_locs'
			if dict_idx in config_params.keys():
				food_locs = config_params[dict_idx]
			else:
				food_locs = [tuple(x) for x in itertools.product(range(field_length), range(field_length))]
		
		fields = []
		rng_gen = np.random.default_rng(RNG_SEED)
		for _ in range(n_runs):
			field = np.zeros((field_length, field_length), np.int32)
			n_spawn_foods = 0
			while n_spawn_foods < args.foods:
				field = spawn_food(field, food_locs, rng_gen, (field_length, field_length), args.foods)
				n_spawn_foods += 1
			fields += [field]
		
		results = {}
		log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs'
		log_prefix = filename_prefix + '_' + str(n_runs) + '_' + ''.join([str(x) for x in team_comps])
		if verbose:
			print('Sarting evals')
		for comp in team_comps:
			avg_steps, avg_pred, run_steps, pred_steps, history = eval_behaviour(n_runs, args.agents, args.foods, env, comp,
																				 (states, actions_sa, transitions_sa, leader_rewards, follower_rewards),
																				 (*leader_opt_decision, *leader_leg_decision),
																				 (*follower_opt_decision, *follower_leg_decision),
																				 (actions_ma, transitions_ma, rewards_ma),
																				 optimal_ma_decision, fields, food_locs, use_render,
																				 data_dir, log_dir, log_prefix, verbose)
			
			results[comp] = {'avg steps': avg_steps, 'run steps': run_steps, 'avg predictions': avg_pred, 'predictions steps': pred_steps, 'history': history}
		
		results_file = data_dir / 'results' / (filename_prefix + '_f' + str(args.foods) + '_' + str(n_runs) + '_' + ''.join([str(x) for x in team_comps]) + '.csv')
		write_full_results_csv(results_file, results, 'w', ['comp', 'avg steps', 'run steps', 'avg predictions', 'predictions steps', 'history'])


if __name__ == '__main__':
	main()
