#! /usr/bin/env python

import numpy as np
import argparse
import itertools
import pickle
import yaml
import scipy

from environments.lb_foraging_plan import MultiAgentForagingPlan, SingleAgentForagingPlan, ForagingPlan
from agents.mdp_agent import MDPAgent
from agents.legible_mdp_agent import LegibleMDPAgent
from termcolor import colored
from typing import Tuple, List, Dict
from tqdm import tqdm
from pathlib import Path


GAMMA = 0.9
BETA = 0.9


def get_goal_states(states: Tuple, goal: Tuple) -> List[int]:
	
	goals = []
	goal_row, goal_col = goal
	goal_adj_locs = [(goal_row + 1, goal_col), (goal_row - 1, goal_col), (goal_row, goal_col + 1), (goal_row, goal_col - 1)]
	
	for state in states:
		
		state_locs, _, food_status = ForagingPlan.get_state_tuple(state)
		
		if state_locs[0] in goal_adj_locs and food_status == 1:
			goals += [states.index(state)]
	
	return goals


def store_decision_model(q: List[np.ndarray], pol: List[np.ndarray], filename: str) -> None:
	
	save_path = Path(__file__).parent.absolute().parent.absolute() / 'models'
	if not save_path.is_dir():
		save_path.mkdir(parents=True, exist_ok=False)
	save_file = save_path / (filename + '.pkl')
	model = {'q_matrix': q, 'pol': pol}
	pickle.dump(model, open(save_file, 'wb'))


def store_env_model(x: Tuple, a: Tuple, p: Dict, rewards: Dict, filename: str) -> None:
	
	save_path = Path(__file__).parent.absolute().parent.absolute() / 'models'
	if not save_path.is_dir():
		save_path.mkdir(parents=True, exist_ok=False)
	save_file = save_path / (filename + '.pkl')
	model = {'states': x, 'actions': a, 'transitions': p, 'rewards': rewards}
	pickle.dump(model, open(save_file, 'wb'))


def main():
	
	parser = argparse.ArgumentParser(description='Legible behavior foraging scenario')
	parser.add_argument('--rows', dest='rows', required=True, type=int, help='Number of rows in environment')
	parser.add_argument('--cols', dest='cols', required=True, type=int, help='Number of cols in environment')
	parser.add_argument('--max_food', dest='max_food', required=True, type=int, help='Maximum number of simultaneous food items in the environment')
	parser.add_argument('--food_level', dest='food_lvl', required=True, type=int, help='Level of food in environment')
	parser.add_argument('--agents', dest='n_agents', required=True, type=int, help='Number of agents in environment')
	parser.add_argument('--agent_levels', dest='agent_lvls', required=True, type=int, nargs='+', help='List with the level for each agent in the environment')
	parser.add_argument('--agent_mode', dest='mode', required=False, type=str, choices=['leader', 'follower'], default='nospec',
						help='The type of agent being generated behaviour: either a leader or a follower. Each type has a social role behavior associated.'
							 ' If no is specified, then a general behavior is created, with no specific social role.')
	
	args = parser.parse_args()
	mode = args.mode
	
	if len(args.agent_lvls) != args.n_agents:
		print(colored('[ERROR] Number of agents different than number of supplied agent levels. Exiting application', 'red'))
		return
	
	filename_prefix = 'lbforaging_' + str(args.rows) + 'x' + str(args.cols) + '_a' + str(args.n_agents) + 'l' + str(args.food_lvl)
	
	data_dir = Path(__file__).parent.absolute().parent.absolute() / 'data'
	with open(data_dir / 'configs' / 'lbforaging_plan_configs.yaml') as file:
		config_params = yaml.full_load(file)
		dict_idx = str(args.rows) + 'x' + str(args.cols) + '_food_locs'
		if dict_idx in config_params.keys():
			food_locs = config_params[dict_idx]
		else:
			food_locs = [item for item in itertools.product(range(args.rows), range(args.cols))]
	
	# Generate world transitions for different food positions (free multi agent setting)
	field_size = (args.rows, args.cols)
	multi_agent_env = MultiAgentForagingPlan(field_size, args.max_food, args.food_lvl, args.n_agents, args.agent_lvls, args.food_lvl)
	for food in tqdm(food_locs, desc='Free multi agent world generation'):
		row, col = food
		multi_agent_env.generate_world_food((row, col, args.food_lvl))
		
	# Store environment
	store_env_model(multi_agent_env.states, multi_agent_env.actions, multi_agent_env.transitions, multi_agent_env.rewards, filename_prefix + '_ma_environment')

	# Compute optimal Q-functions and policies
	ma_pol_opt = []
	ma_Q_opt = []
	for key in tqdm(multi_agent_env.transitions.keys(), desc='Free multi agent optimal decision computation'):
		mdp = MDPAgent(multi_agent_env.states, multi_agent_env.actions, [multi_agent_env.transitions[key]], multi_agent_env.rewards[key],
					   GAMMA, 'rewards', False)
		pol, q = mdp.policy_iteration()
		ma_pol_opt += [pol]
		ma_Q_opt += [q]
	
	# Store optimal Q-functions and policies
	store_decision_model(ma_Q_opt, ma_pol_opt, filename_prefix + '_ma_optimal_decision')

	if mode == 'leader':
		social_roles = ['up', 'down']
	elif mode == 'follower':
		social_roles = ['left', 'right']
	else:
		social_roles = ['up', 'down', 'left', 'right']

	# Generate world transitions for different food positions (optimal multi agent setting)
	opt_multi_agent_env = SingleAgentForagingPlan(field_size, args.max_food, args.food_lvl, args.n_agents, args.agent_lvls, ma_pol_opt,
												  multi_agent_env.actions, food_locs, social_roles)
	# print(opt_multi_agent_env.food_pos_lst, food_locs)
	for food in tqdm(food_locs, desc='Optimal multi agent world generation'):
		row, col = food
		opt_multi_agent_env.generate_world_food((row, col, args.food_lvl))

	# Store environment
	store_env_model(opt_multi_agent_env.states, opt_multi_agent_env.actions, opt_multi_agent_env.transitions,
					opt_multi_agent_env.rewards, filename_prefix + '_' + mode + '_environment')

	# Compute optimal Q-functions and policies
	Q_opt = []
	pol_opt = []
	opt_multi_agent_env_transitions = []
	for key in tqdm(opt_multi_agent_env.transitions.keys(), desc='Optimal multi agent world optimal decision computation'):
		mdp = MDPAgent(opt_multi_agent_env.states, opt_multi_agent_env.actions, [opt_multi_agent_env.transitions[key]], opt_multi_agent_env.rewards[key],
					   GAMMA, 'rewards', False)
		pol, q = mdp.policy_iteration()
		Q_opt += [q]
		pol_opt += [pol]
		opt_multi_agent_env_transitions += [opt_multi_agent_env.transitions[key]]

	# Store optimal Q-functions and policies
	store_decision_model(Q_opt, pol_opt, filename_prefix + '_' + mode + '_optimal_decision')

	# Compute legible Q-functions and policies
	legible_mdp = LegibleMDPAgent(opt_multi_agent_env.states, opt_multi_agent_env.actions, opt_multi_agent_env_transitions, GAMMA, False,
								  list(opt_multi_agent_env.transitions.keys()), BETA, 1, Q_opt)
	# legible_rewards = legible_mdp.costs
	# for state_idx in range(len(opt_multi_agent_env.states)):
	# 	print(opt_multi_agent_env.states[state_idx] + str(legible_rewards[0][state_idx]))
	Q_legible = []
	pol_legible = []
	for obj in tqdm(range(len(opt_multi_agent_env.transitions.keys())), desc='Optimal multi agent world legible decision computation'):
		pol, q = legible_mdp.policy_iteration(obj)
		Q_legible += [q]
		pol_legible += [pol]

	# Store legible Q-functions and policies
	store_decision_model(Q_legible, pol_legible, filename_prefix + '_' + mode + '_legible_decision')
	

if __name__ == '__main__':
	main()
