#! /usr/bin/env python

import numpy as np
import timeit
import argparse
import yaml
np.set_printoptions(precision=5)

from tqdm import tqdm
from mdp import LegibleTaskMDP, MiuraLegibleMDP, MDP, Utilities
from mazeworld import SimpleWallMazeWorld2
from typing import Dict, List, Tuple
from termcolor import colored

# SCALABILITY_WORLDS = {1: '5x5_world.yaml', 2: '10x10_world_2.yaml', 3: '25x25_world.yaml',
# 					  4: '50x50_world.yaml', 5: '75x75_world.yaml', 6: '100x100_world.yaml',
# 					  7: '150x150_world.yaml', 8: '200x200_world.yaml'}
SCALABILITY_WORLDS = {2: '10x10_world_2.yaml'}

OBJECTS_WORLDS = {1: '100x100_world.yaml', 2: '100x100_world_2.yaml', 3: '100x100_world_3.yaml',
				  4: '100x100_world_4.yaml', 5: '100x100_world_5.yaml', 6: '100x100_world_6.yaml'}


def wrapper(func, *args, **kwargs):
	def wrapped():
		return func(*args, **kwargs)
	return wrapped


def get_goal_states(states: np.ndarray, goal: str) -> List[int]:
	state_lst = list(states)
	return [state_lst.index(state) for state in states if state.find(goal) != -1]


def scalability_eval(n_reps: int, fail_prob: float, beta: float, gamma: float) -> Tuple[Dict, Dict]:

	def policy_trajectory(task_mdp_w, tasks, goal, x0):
		task_pol_w, _ = task_mdp_w.policy_iteration(tasks.index(goal))
		task_traj, _ = task_mdp_w.trajectory(x0, task_pol_w)
		return task_traj
		
	def miura_trajectory(mdp, miura_mdp, x0, depth, n_its, beta):
		pol_w, _ = mdp.policy_iteration()
		miura_traj = miura_mdp.legible_trajectory(x0, pol_w, depth, n_its, beta)
		return miura_traj

	worlds_keys = SCALABILITY_WORLDS.keys()
	miura_times = {}
	policy_times = {}
	for _ in tqdm(range(n_reps), desc='Scalability Repetitions'):

		for idx in worlds_keys:
		
			# Load mazeworld information
			with open('../data/configs/' + SCALABILITY_WORLDS[idx]) as file:
				config_params = yaml.full_load(file)
				
				n_cols = config_params['n_cols']
				n_rows = config_params['n_rows']
				walls = config_params['walls']
				task_states = config_params['task_states']
				tasks = config_params['tasks']
				
			# Setup mazeworld, mdps goal and goal states
			swmw = SimpleWallMazeWorld2()
			X_w, A_w, P_w = swmw.generate_world(n_rows, n_cols, task_states, walls, 'stochastic', fail_prob)
			nX = len(X_w)
			nA = len(A_w)
			goal = np.random.choice(tasks)
			goal_states = get_goal_states(X_w, goal)
			
			# Create MDPs
			mdps_w = {}
			v_mdps_w = {}
			q_mdps_w = {}
			task_mdps_w = {}
			dists = []
			costs = []
			rewards = {}
			for i in range(len(tasks)):
				c = swmw.generate_rewards(tasks[i], X_w, A_w)
				costs += [c]
				rewards[tasks[i]] = c
				mdp = MDP(X_w, A_w, P_w, c, gamma, get_goal_states(X_w, tasks[i]), 'rewards')
				pol, q = mdp.policy_iteration()
				v = Utilities.v_from_q(q, pol)
				q_mdps_w[tasks[i]] = q
				v_mdps_w[tasks[i]] = v
				mdps_w['mdp' + str(i + 1)] = mdp
			dists = np.array(dists)
			
			# Create each framework's Legible MDPs
			policy_mdp = LegibleTaskMDP(X_w, A_w, P_w, gamma, goal, task_states, tasks, beta, goal_states, 1,
										'leg_optimal', q_mdps=q_mdps_w, v_mdps=v_mdps_w, dists=dists)
			miura_mdp = MiuraLegibleMDP(X_w, A_w, P_w, 0.9, goal, tasks, beta, goal_states, q_mdps=q_mdps_w)
			
			# Find possible initial states
			nonzerostates = np.nonzero(q_mdps_w[goal].sum(axis=1))[0]
			init_states = [np.delete(nonzerostates, np.argwhere(nonzerostates == g)) for g in goal_states][0]
			x0 = X_w[np.random.choice(init_states)]
			
			# Wrappers to run timeit on the trajectories for each framework
			policy_stmt = wrapper(policy_trajectory, policy_mdp, tasks, goal, x0)
			miura_stmt = wrapper(miura_trajectory, mdps_w['mdp' + str(tasks.index(goal) + 1)], miura_mdp, x0, 20, 50000, beta)
			
			# Time each framework's performance and store it
			policy_time = timeit.timeit(policy_stmt, number=10)
			miura_time = timeit.timeit(miura_stmt, number=10)
			if nX not in policy_times:
				policy_times[nX] = policy_time / n_reps
			else:
				policy_times[nX] += policy_time / n_reps
			if nX not in miura_times:
				miura_times[nX] = miura_time / n_reps
			else:
				miura_times[nX] += miura_time / n_reps

	return policy_times, miura_times


def main():
	
	parser = argparse.ArgumentParser(description='Legibility performance evaluation comparison between frameworks')
	parser.add_argument('--evaluation', dest='evaluation_type', type=str, required=True, choices=['scale', 'goals'],
						help='Evaluation method to use.')
	parser.add_argument('--metric', dest='metric', type=str, required=True, choices=['shiura', 'policy'],
						help='Metric to compare legibility performances.')
	parser.add_argument('--reps', dest='reps', type=int, required=True,
						help='Number of repetitions for the evaluation cycle for each framework.')
	parser.add_argument('--fail_prob', dest='fail_prob', type=float, required=True,
						help='Probability of movement failing.')
	parser.add_argument('--beta', dest='beta', type=float, required=True,
						help='Constant that guides how close to the optimal policy the legbility is.')
	parser.add_argument('--gamma', dest='gamma', type=float, required=True,
						help='Discount factor for the MDPs')
	# parser.add_argument('--name', dest='', type=, required=, choices=[],
	# 					help='')
	
	args = parser.parse_args()
	evaluation = args.evaluation_type
	metric = args.metric
	n_reps = args.reps
	fail_prob = args.fail_prob
	beta = args.beta
	gamma = args.gamma
	
	if evaluation == 'scale':
		scalability_times = scalability_eval(n_reps, fail_prob, beta, gamma)
		print('Policy times: ' + str(scalability_times))
	elif evaluation == 'goals':
		pass
	
	else:
		print(colored('[ERROR] Invalid evaluation type, exiting program!', color='red'))
		return


if __name__ == '__main__':
	main()
