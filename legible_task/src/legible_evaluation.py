#! /usr/bin/env python
import sys
import time

import numpy as np
import timeit
import argparse
import yaml
import csv
np.set_printoptions(precision=5)

from tqdm import tqdm
from mdp import LegibleTaskMDP, MiuraLegibleMDP, MDP, Utilities
from mazeworld import SimpleWallMazeWorld2
from typing import Dict, List, Tuple
from termcolor import colored

SCALABILITY_WORLDS = {1: '10x10_world_2.yaml', 2: '25x25_world.yaml', 3: '50x50_world.yaml', 4: '75x75_world.yaml', 5: '100x100_world.yaml'}
# 						, 6: '125x125_world.yaml', 7: '150x150_world.yaml'}

OBJECTS_WORLDS = {1: '100x100_world.yaml', 2: '100x100_world_2.yaml', 3: '100x100_world_3.yaml',
				  4: '100x100_world_4.yaml', 5: '100x100_world_5.yaml', 6: '100x100_world_6.yaml'}


class IncorrectEvaluationTypeError(Exception):

	def __init__(self, eval_type, message="Evaluation type is not in list [scale, goals]"):
		self.eval_type = eval_type
		self.message = message
		super().__init__(self.message)

	def __str__(self):
		return f'{self.eval_type} -> {self.message}'


class InvalidEvaluationMetric(Exception):

	def __init__(self, metric, message="Chosen evaluation metric is not among the possibilities: miura, policy or time"):
		self.metric = metric
		self.message = message
		super().__init__(self.message)

	def __str__(self):
		return f'{self.metric} -> {self.message}'


def wrapper(func, *args, **kwargs):
	def wrapped():
		return func(*args, **kwargs)
	return wrapped


def get_goal_states(states: np.ndarray, goal: str) -> List[int]:
	state_lst = list(states)
	return [state_lst.index(state) for state in states if state.find(goal) != -1]


def frameworks_evaluation(evaluation: str, n_reps: int, fail_prob: float, beta: float, gamma: float, metric: str) -> Tuple[Dict, Dict]:

	# Auxiliary methods to test the performance of each framework in obtaining a sequence of legible actions
	def policy_trajectory(task_mdp_w, tasks, goal, x0) -> Tuple[np.ndarray, np.ndarray]:
		task_pol_w, _ = task_mdp_w.policy_iteration(tasks.index(goal))
		task_traj = task_mdp_w.trajectory(x0, task_pol_w)
		return task_traj
		
	def miura_trajectory(mdp, miura_mdp, x0, depth, n_its, beta) -> Tuple[np.ndarray, np.ndarray]:
		pol_w, _ = mdp.policy_iteration()
		miura_traj = miura_mdp.legible_trajectory(x0, pol_w, depth, n_its, beta)
		return miura_traj

	# Auxiliary methods to evaluate the legibility performance of each framework
	def policy_evaluation(trajectory: Tuple[np.ndarray, np.ndarray], task_idx: int, policy_mdp: LegibleTaskMDP) -> float:
		
		trajs = np.array([trajectory])
		return policy_mdp.trajectory_reward(trajs, task_idx)
	
	def miura_evaluation(trajectory: Tuple[np.ndarray, np.ndarray], task_idx: int, miura_mdp: MiuraLegibleMDP):
		
		trajs = np.array([trajectory])
		return miura_mdp.trajectory_reward(trajs, task_idx)

	miura_eval = {}
	policy_eval = {}
	if evaluation == 'scale':
		worlds_keys = SCALABILITY_WORLDS.keys()
	elif evaluation == 'goals':
		worlds_keys = OBJECTS_WORLDS.keys()
	else:
		raise IncorrectEvaluationTypeError(evaluation)
	
	print('Started Evaluation process')
	for idx in worlds_keys:
	
		print('Starting evaluation for world: ' + str(idx))
	
		# Load mazeworld information
		with open('../data/configs/' + SCALABILITY_WORLDS[idx]) as file:
			config_params = yaml.full_load(file)
			
			n_cols = config_params['n_cols']
			n_rows = config_params['n_rows']
			walls = config_params['walls']
			task_states = config_params['task_states']
			tasks = config_params['tasks']
		
		# Setup mazeworld
		swmw = SimpleWallMazeWorld2()
		X_w, A_w, P_w = swmw.generate_world(n_rows, n_cols, task_states, walls, 'stochastic', fail_prob)
		nX = len(X_w)
		nA = len(A_w)
		
		for iteration in tqdm(range(n_reps), desc='Evaluation Repetitions'):
			
			# Choose current goal and obtain the corresponding goal states
			goal = np.random.choice(tasks)
			goal_states = get_goal_states(X_w, goal)
			
			# Create MDPs
			mdps_w = {}
			v_mdps_w = {}
			q_mdps_w = {}
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
				
			if metric == 'miura':
				# Obtain sample trajectory with each framework
				policy_traj = policy_trajectory(policy_mdp, tasks, goal, x0)
				miura_traj = miura_trajectory(mdps_w['mdp' + str(tasks.index(goal) + 1)], miura_mdp, x0, 20, 50000, beta)
				
				# Performance evaluation according to the metric in Miura et al. and storing
				policy_performance = miura_evaluation(policy_traj, tasks.index(goal), miura_mdp)
				miura_performance = miura_evaluation(miura_traj, tasks.index(goal), miura_mdp)
				if nX not in policy_eval:
					policy_eval[nX] = policy_performance / n_reps
				else:
					policy_eval[nX] += policy_performance / n_reps
				if nX not in miura_eval:
					miura_eval[nX] = miura_performance / n_reps
				else:
					miura_eval[nX] += miura_performance / n_reps
				
			elif metric == 'policy':
				# Obtain sample trajectory with each framework
				policy_traj = policy_trajectory(policy_mdp, tasks, goal, x0)
				miura_traj = miura_trajectory(mdps_w['mdp' + str(tasks.index(goal) + 1)], miura_mdp, x0, 20, 50000, beta)
				
				# Performance evaluation according to the metric of policy legibility and storing
				policy_performance = policy_evaluation(policy_traj, tasks.index(goal), policy_mdp)
				miura_performance = policy_evaluation(miura_traj, tasks.index(goal), policy_mdp)
				if nX not in policy_eval:
					policy_eval[nX] = policy_performance / n_reps
				else:
					policy_eval[nX] += policy_performance / n_reps
				if nX not in miura_eval:
					miura_eval[nX] = miura_performance / n_reps
				else:
					miura_eval[nX] += miura_performance / n_reps
			
			elif metric == 'time':
				# Wrappers to run timeit on the trajectories for each framework
				policy_stmt = wrapper(policy_trajectory, policy_mdp, tasks, goal, x0)
				miura_stmt = wrapper(miura_trajectory, mdps_w['mdp' + str(tasks.index(goal) + 1)], miura_mdp, x0, 20, 50000, beta)
				
				# Time each framework's performance and store it
				policy_time = timeit.timeit(policy_stmt, number=10)
				miura_time = timeit.timeit(miura_stmt, number=10)
				if nX not in policy_eval:
					policy_eval[nX] = policy_time / n_reps
				else:
					policy_eval[nX] += policy_time / n_reps
				if nX not in miura_eval:
					miura_eval[nX] = miura_time / n_reps
				else:
					miura_eval[nX] += miura_time / n_reps
			else:
				raise InvalidEvaluationMetric(metric)

			if iteration % 100 == 0:
				print('Reached iteration %d!! Completed %.2f%% of the repetitions\n\n' % (iteration, iteration / n_reps * 100))

	return policy_eval, miura_eval


def main():
	
	parser = argparse.ArgumentParser(description='Legibility performance evaluation comparison between frameworks')
	parser.add_argument('--evaluation', dest='evaluation_type', type=str, required=True, choices=['scale', 'goals'],
						help='Evaluation method to use.')
	parser.add_argument('--metric', dest='metric', type=str, required=True, choices=['miura', 'policy', 'time'],
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
	
	try:
		print('Start time: ' + str(time.ctime()))
		policy_results, miura_results = frameworks_evaluation(evaluation, n_reps, fail_prob, beta, gamma, metric)
		results = {'policy': policy_results, 'miura': miura_results}
		
		fields = ['framework'] + list(policy_results.keys())
		csv_file = '../data/results/evaluation_results_' + evaluation + '_' + metric + '.csv'
		try:
			with open(csv_file, 'w') as csvfile:
				writer = csv.DictWriter(csvfile, fieldnames=fields)
				writer.writeheader()
				for key, val in sorted(results.items()):
					row = {'framework': key}
					row.update(val)
					writer.writerow(row)
		
		except IOError as e:
			print("I/O error: " + str(e))
		print('End time: ' + str(time.ctime()))
	
	except IncorrectEvaluationTypeError:
		print(colored('[ERROR] Invalid evaluation type, exiting program!', color='red'))
		raise
	
	except InvalidEvaluationMetric:
		print(colored('[ERROR] Invalid metric, exiting program!', color='red'))
		raise
	

if __name__ == '__main__':
	main()
