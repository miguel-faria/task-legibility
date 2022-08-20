#! /usr/bin/env python
import sys

import numpy as np
import argparse
import itertools
import pickle
import yaml
import scipy

from environments.lb_foraging_plan_mb import MultiAgentMBForagingPlan, MBForagingPlan, FoodStates, AgentRoles
from agents.mdp_agent import MDPAgent
from agents.legible_mdp_agent import LegibleMDPAgent
from termcolor import colored
from typing import Tuple, List, Dict
from tqdm import tqdm
from pathlib import Path
from policies import Policies

GAMMA = 0.9
BETA = 0.9
RNG_SEED = 20220725
EXPLORE = 0.75


def get_goal_states(states: Tuple, goal: Tuple) -> List[int]:
	goals = []
	goal_row, goal_col = goal
	goal_adj_locs = [(goal_row + 1, goal_col), (goal_row - 1, goal_col), (goal_row, goal_col + 1), (goal_row, goal_col - 1)]
	
	for state in states:
		
		agent_loc, lvl, rel_pos, food_status = MBForagingPlan.get_state_tuple(state)
		
		if agent_loc in goal_adj_locs and food_status == FoodStates.PICKED:
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


def main( ):
	parser = argparse.ArgumentParser(description='Legible behavior foraging scenario')
	parser.add_argument('--rows', dest='rows', required=True, type=int, help='Number of rows in environment')
	parser.add_argument('--cols', dest='cols', required=True, type=int, help='Number of cols in environment')
	parser.add_argument('--max_food', dest='max_food', required=True, type=int, help='Maximum number of simultaneous food items in the environment')
	parser.add_argument('--food_level', dest='food_lvl', required=True, type=int, help='Level of food in environment')
	parser.add_argument('--agents', dest='n_agents', required=True, type=int, help='Number of agents in environment')
	parser.add_argument('--agent_levels', dest='agent_lvls', required=True, type=int, nargs='+', help='List with the level for each agent in the environment')
	parser.add_argument('--agent_roles', dest='agent_roles', required=True, type=int, nargs='+',
						help='List with the social roles for each agent in the environment:'
							 '\n\t0 - No specific role, can pick food either laterally or vertically'
							 '\n\t1 - Leader role, agent can pick food only from above or below the food item'
							 '\n\t2 - Follower role, agent can pick food only from the left or right of the food item')
	parser.add_argument('--max_steps', dest='max_steps', required=True, type=int, help='Maximum number of steps for Model Base to run')
	parser.add_argument('--finish', dest='finish', action='store_true', help='Flag that signals if model base starts from beginning or from a previous savepoint')
	
	args = parser.parse_args()
	rows = args.rows
	cols = args.cols
	field_size = (rows, cols)
	food_lvl = args.food_lvl
	n_food_max = args.max_food
	n_agents = args.n_agents
	agent_lvls = args.agent_lvls
	roles = args.agent_roles
	max_steps = args.max_steps
	finish_mb = args.finish
	
	log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs'
	filename_prefix = 'lbforaging_mb_' + str(rows) + 'x' + str(cols) + '_a' + str(n_agents) + 'l' + str(food_lvl)
	sys.stdout = open(log_dir / (filename_prefix + '_log.txt'), 'w')
	sys.stderr = open(log_dir / (filename_prefix + '_err.txt'), 'w')
	
	if len(agent_lvls) != n_agents:
		print(colored('[ERROR] Number of agents different than number of supplied agent levels. Exiting application', 'red'))
		return
	
	if len(roles) != n_agents:
		print(colored('[ERROR] Number of agents different than number of agent roles. Exiting application', 'red'))
		return
	
	data_dir = Path(__file__).parent.absolute().parent.absolute() / 'data'
	with open(data_dir / 'configs' / 'lbforaging_plan_configs.yaml') as file:
		config_params = yaml.full_load(file)
		dict_idx = str(rows) + 'x' + str(cols) + '_food_locs'
		if dict_idx in config_params.keys():
			food_locs = config_params[dict_idx]
		else:
			food_locs = [item for item in itertools.product(range(rows), range(cols))]
	
	# Generate world transitions for different food positions (free multi agent setting)
	print('Generating multi-agent model based environment, transitions and Q*')
	multi_agent_env = MultiAgentMBForagingPlan(field_size, n_food_max, food_lvl, n_agents, agent_lvls, roles, food_lvl)
	for food in tqdm(food_locs, desc='Joint multi agent world generation'):
		row, col = food
		rng_gen = np.random.default_rng(RNG_SEED)
		multi_agent_env.generate_world_food((row, col, food_lvl), Policies.eps_greedy, rng_gen, EXPLORE, max_steps, GAMMA, finish_mb)
	
	# Store environment
	store_env_model(multi_agent_env.states, multi_agent_env.actions, multi_agent_env.transitions, multi_agent_env.rewards, filename_prefix + '_ma_environment')
	
	# Compute optimal Q-functions and policies
	ma_pol_opt = []
	ma_Q_opt = []
	for key in tqdm(multi_agent_env.transitions.keys(), desc='Joint multi agent optimal decision computation'):
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
