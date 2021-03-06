#! /usr/bin/env python
import sys
import os
import time
import signal

import numpy as np
import timeit
import argparse
import yaml
import csv
np.set_printoptions(precision=5)

from tqdm import tqdm
from mdp import LegibleTaskMDP, MiuraLegibleMDP, MDP, Utilities
from mazeworld import SimpleWallMazeWorld2
from utilities import InvalidEvaluationTypeError, InvalidEvaluationMetricError, InvalidFrameworkError, TimeoutException, store_savepoint, load_savepoint, signal_handler
from typing import Dict, List, Tuple
from termcolor import colored
from pathlib import Path
from multiprocessing import Process

SCALABILITY_WORLDS = {1: '10x10_world_2', 2: '25x25_world',  3: '40x40_world', 4: '50x50_world',
					  5: '60x60_world', 6: '75x75_world', 7: '5x8_world'}

OBJECTS_WORLDS = {1: '25x25_world_g3', 2: '25x25_world_g4', 3: '25x25_world_g5', 4: '25x25_world_g6',
				  5: '25x25_world_g7', 6: '25x25_world_g8', 7: '25x25_world_g9', 8: '25x25_world_g10'}


def write_iterations_results_csv(csv_file: Path, results: Dict, access_type: str, fields: List[str], iteration_data: Tuple, n_iteration: int) -> None:
	try:
		with open(csv_file, access_type) as csvfile:
			field_names = ['iteration_test'] + fields
			writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',', lineterminator='\n')
			if access_type != 'a':
				writer.writeheader()
			it_idx = str(n_iteration) + ' ' + ' '.join(iteration_data)
			row = {'iteration_test': it_idx}
			row.update(results.items())
			writer.writerow(row)
	
	except IOError as e:
		print(colored("I/O error: " + str(e), color='red'))


def write_full_results_csv(csv_file: Path, metric: str, results: Dict, access_type: str, fields: List[str]) -> None:
	try:
		print('Results to write: ' + str(list(results.items())))
		with open(csv_file, access_type) as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=fields, delimiter=',', lineterminator='\n')
			if access_type != 'a':
				writer.writeheader()
			for key, val in sorted(results.items()):
				row = {'world_size': key}
				row.update(val)
				writer.writerow(row)
	
	except IOError as e:
		print(colored("I/O error: " + str(e), color='red'))
		

def wrapper(func, *args, **kwargs):
	def wrapped():
		return func(*args, **kwargs)
	return wrapped


def get_goal_states(states: np.ndarray, goal: str) -> List[int]:
	state_lst = list(states)
	return [state_lst.index(state) for state in states if state.find(goal) != -1]


def world_iteration(states: np.ndarray, actions: List[str], transitions: Dict, beta: float, evaluation: str, gamma: float, metric: str,
					n_tasks: int, n_states: int, n_reps: int, eval_results: Dict, task_states: List[Tuple], tasks: List[str],
					framework: str, verbose: bool, state_goal: Tuple, q_mdps: Dict, v_mdps: Dict, mdps: Dict, dists: np.ndarray,
					data_dir: Path, n_iteration: int, world: str) -> None:
	
	# Auxiliary methods to test the performance of each framework in obtaining a sequence of legible actions
	def policy_trajectory(task_mdp_w: LegibleTaskMDP, tasks: List[str], goal: str, x0: str, rng_gen: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
		task_pol_w, _ = task_mdp_w.policy_iteration(tasks.index(goal))
		task_traj = task_mdp_w.trajectory(x0, task_pol_w, rng_gen)
		return task_traj
	
	def miura_trajectory(mdp: MDP, miura_mdp: MiuraLegibleMDP, x0: str, depth: int, n_its: int,
						 beta: float, verbose: bool, rng_gen: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
		pol_w, _ = mdp.policy_iteration()
		miura_traj = miura_mdp.legible_trajectory(x0, pol_w, depth, n_its, beta, verbose, rng_gen)
		return miura_traj
	
	# Auxiliary methods to evaluate the legibility performance of each framework
	def policy_evaluation(trajectory: Tuple[np.ndarray, np.ndarray], task_idx: int, policy_mdp: LegibleTaskMDP) -> float:
		trajs = np.array([trajectory])
		return policy_mdp.trajectory_reward(trajs, task_idx)
	
	def miura_evaluation(trajectory: Tuple[np.ndarray, np.ndarray], task_idx: int, miura_mdp: MiuraLegibleMDP) -> float:
		trajs = np.array([trajectory])
		return miura_mdp.trajectory_reward(trajs, task_idx)
	
	# Choose current goal and obtain the corresponding goal states
	rng_gen = np.random.default_rng(2021)
	x0 = state_goal[0]
	goal = state_goal[1]
	goal_states = get_goal_states(states, goal)
	
	# Create each framework's Legible MDPs
	policy_mdp = LegibleTaskMDP(states, actions, transitions, gamma, verbose, goal, task_states, tasks, beta, goal_states, 1,
								'leg_optimal', q_mdps=q_mdps, v_mdps=v_mdps, dists=dists)
	miura_mdp = MiuraLegibleMDP(states, actions, transitions, gamma, verbose, goal, tasks, beta, goal_states, q_mdps=q_mdps)
	
	print('Starting state: %s\tGoal: %s' % (x0, goal))
	sys.stdout.flush()
	sys.stderr.flush()
	signal.signal(signal.SIGALRM, signal_handler)
	signal.alarm(7200)  # Two hours to complete evaluation
	results = {}
	try:
		if framework == 'policy':
			if metric == 'all':
				# Obtain sample trajectory with each framework
				policy_traj = policy_trajectory(policy_mdp, tasks, goal, x0, rng_gen)
				
				# Wrappers to run timeit on the trajectories for each framework
				policy_stmt = wrapper(policy_trajectory, policy_mdp, tasks, goal, x0, rng_gen)
				
				# Performance evaluation
				policy_m_performance = miura_evaluation(policy_traj, tasks.index(goal), miura_mdp)
				policy_p_performance = policy_evaluation(policy_traj, tasks.index(goal), policy_mdp)
				policy_t_performance = timeit.timeit(policy_stmt, number=1)
				
				results['failures'] = 0
				results['policy'] = policy_p_performance
				results['miura'] = policy_m_performance
				results['time'] = policy_t_performance
				
				if evaluation == 'scale':
					if n_states not in eval_results:
						eval_results[str(n_states)] = dict()
						eval_results[str(n_states)]['failures'] = 0
						eval_results[str(n_states)]['policy'] = policy_p_performance / n_reps
						eval_results[str(n_states)]['miura'] = policy_m_performance / n_reps
						eval_results[str(n_states)]['time'] = policy_t_performance / n_reps
					else:
						eval_results[str(n_states)]['policy'] += policy_p_performance / n_reps
						eval_results[str(n_states)]['miura'] += policy_m_performance / n_reps
						eval_results[str(n_states)]['time'] += policy_t_performance / n_reps
				
				elif evaluation == 'goals':
					if n_tasks not in eval_results:
						eval_results[str(n_tasks)] = dict()
						eval_results[str(n_tasks)]['failures'] = 0
						eval_results[str(n_tasks)]['policy'] = policy_p_performance / n_reps
						eval_results[str(n_tasks)]['miura'] = policy_m_performance / n_reps
						eval_results[str(n_tasks)]['time'] = policy_t_performance / n_reps
					else:
						eval_results[str(n_tasks)]['policy'] += policy_p_performance / n_reps
						eval_results[str(n_tasks)]['miura'] += policy_m_performance / n_reps
						eval_results[str(n_tasks)]['time'] += policy_t_performance / n_reps
				
				else:
					raise InvalidEvaluationTypeError(evaluation)
			
			else:
				if metric == 'miura':
					# Obtain sample trajectory with each framework
					policy_traj = policy_trajectory(policy_mdp, tasks, goal, x0, rng_gen)
					
					# Performance evaluation according to the metric in Miura et al. and storing
					policy_performance = miura_evaluation(policy_traj, tasks.index(goal), miura_mdp)
				
				elif metric == 'policy':
					# Obtain sample trajectory with each framework
					policy_traj = policy_trajectory(policy_mdp, tasks, goal, x0, rng_gen)
					
					# Performance evaluation according to the metric of policy legibility and storing
					policy_performance = policy_evaluation(policy_traj, tasks.index(goal), policy_mdp)
				
				elif metric == 'time':
					# Wrappers to run timeit on the trajectories for each framework
					policy_stmt = wrapper(policy_trajectory, policy_mdp, tasks, goal, x0, rng_gen)
					
					# Time each framework's performance and store it
					policy_performance = timeit.timeit(policy_stmt, number=1)
				
				else:
					raise InvalidEvaluationMetricError(metric)
				
				results['failures'] = 0
				results[metric] = policy_performance
				
				if evaluation == 'scale':
					if n_states not in eval_results:
						eval_results[str(n_states)] = dict()
						eval_results[str(n_states)]['failures'] = 0
						eval_results[str(n_states)][metric] = policy_performance / n_reps
					else:
						eval_results[str(n_states)][metric] += policy_performance / n_reps
				
				elif evaluation == 'goals':
					if n_tasks not in eval_results:
						eval_results[str(n_tasks)] = dict()
						eval_results[str(n_tasks)]['failures'] = 0
						eval_results[str(n_tasks)][metric] = policy_performance / n_reps
					else:
						eval_results[str(n_tasks)][metric] += policy_performance / n_reps
				
				else:
					raise InvalidEvaluationTypeError(evaluation)
		
		elif framework == 'miura':
			if metric == 'all':
				# Obtain sample trajectory with each framework
				miura_traj = miura_trajectory(mdps['mdp' + str(tasks.index(goal) + 1)], miura_mdp, x0, 10, 10000, beta, verbose, rng_gen)
				
				# Wrappers to run timeit on the trajectories for each framework
				miura_stmt = wrapper(miura_trajectory, mdps['mdp' + str(tasks.index(goal) + 1)], miura_mdp, x0, 10, 10000, beta, verbose, rng_gen)
				
				# Performance evaluation
				miura_m_performance = miura_evaluation(miura_traj, tasks.index(goal), miura_mdp)
				miura_p_performance = policy_evaluation(miura_traj, tasks.index(goal), policy_mdp)
				miura_t_performance = timeit.timeit(miura_stmt, number=1)
				
				results['failures'] = 0
				results['policy'] = miura_p_performance
				results['miura'] = miura_m_performance
				results['time'] = miura_t_performance
				
				if evaluation == 'scale':
					if n_states not in eval_results:
						eval_results[str(n_states)] = dict()
						eval_results[str(n_states)]['failures'] = 0
						eval_results[str(n_states)]['policy'] = miura_p_performance / n_reps
						eval_results[str(n_states)]['miura'] = miura_m_performance / n_reps
						eval_results[str(n_states)]['time'] = miura_t_performance / n_reps
					else:
						eval_results[str(n_states)]['policy'] += miura_p_performance / n_reps
						eval_results[str(n_states)]['miura'] += miura_m_performance / n_reps
						eval_results[str(n_states)]['time'] += miura_t_performance / n_reps
					
				elif evaluation == 'goals':
					if n_tasks not in eval_results:
						eval_results[str(n_tasks)] = dict()
						eval_results[str(n_tasks)]['failures'] = 0
						eval_results[str(n_tasks)]['policy'] = miura_p_performance / n_reps
						eval_results[str(n_tasks)]['miura'] = miura_m_performance / n_reps
						eval_results[str(n_tasks)]['time'] = miura_t_performance / n_reps
					else:
						eval_results[str(n_tasks)]['policy'] += miura_p_performance / n_reps
						eval_results[str(n_tasks)]['miura'] += miura_m_performance / n_reps
						eval_results[str(n_tasks)]['time'] += miura_t_performance / n_reps
				
				else:
					raise InvalidEvaluationTypeError(evaluation)
			
			else:
				if metric == 'miura':
					# Obtain sample trajectory with each framework
					miura_traj = miura_trajectory(mdps['mdp' + str(tasks.index(goal) + 1)], miura_mdp, x0, 10, 10000, beta, verbose, rng_gen)
					
					# Performance evaluation according to the metric in Miura et al. and storing
					miura_performance = miura_evaluation(miura_traj, tasks.index(goal), miura_mdp)
				
				elif metric == 'policy':
					# Obtain sample trajectory with each framework
					miura_traj = miura_trajectory(mdps['mdp' + str(tasks.index(goal) + 1)], miura_mdp, x0, 10, 10000, beta, verbose, rng_gen)
					
					# Performance evaluation according to the metric of policy legibility and storing
					miura_performance = policy_evaluation(miura_traj, tasks.index(goal), policy_mdp)
				
				elif metric == 'time':
					# Wrappers to run timeit on the trajectories for each framework
					miura_stmt = wrapper(miura_trajectory, mdps['mdp' + str(tasks.index(goal) + 1)], miura_mdp, x0, 10, 10000, beta, verbose, rng_gen)
					
					# Time each framework's performance and store it
					miura_performance = timeit.timeit(miura_stmt, number=1)
				
				else:
					raise InvalidEvaluationMetricError(metric)
				
				results['failures'] = 0
				results[metric] = miura_performance
				
				if evaluation == 'scale':
					if n_states not in eval_results:
						eval_results[str(n_states)] = dict()
						eval_results[str(n_states)]['failures'] = 0
						eval_results[str(n_states)][metric] = miura_performance / n_reps
					else:
						eval_results[str(n_states)][metric] += miura_performance / n_reps
				
				elif evaluation == 'goals':
					if n_tasks not in eval_results:
						eval_results[str(n_tasks)] = dict()
						eval_results[str(n_tasks)]['failures'] = 0
						eval_results[str(n_tasks)][metric] = miura_performance / n_reps
					else:
						eval_results[str(n_tasks)][metric] += miura_performance / n_reps
				
				else:
					raise InvalidEvaluationTypeError(evaluation)
		
		else:
			raise InvalidFrameworkError(framework)
	
	# If takes more than 2 horas to evaluate iteration
	except TimeoutException:
		
		if metric == 'all':
			
			results['failures'] = 1
			results['policy'] = 0
			results['miura'] = 0
			results['time'] = 7200
			
			if evaluation == 'scale':
				if n_states not in eval_results:
					eval_results[str(n_states)] = dict()
					eval_results[str(n_states)]['failures'] = 1
					eval_results[str(n_states)]['policy'] = 0
					eval_results[str(n_states)]['miura'] = 0
					eval_results[str(n_states)]['time'] = 7200 / n_reps
				else:
					eval_results[str(n_states)]['failures'] += 1
					eval_results[str(n_states)]['time'] += 7200 / n_reps
			
			elif evaluation == 'goals':
				if n_tasks not in eval_results:
					eval_results[str(n_tasks)] = dict()
					eval_results[str(n_tasks)]['failures'] = 1
					eval_results[str(n_tasks)]['policy'] = 0
					eval_results[str(n_tasks)]['miura'] = 0
					eval_results[str(n_tasks)]['time'] = 7200 / n_reps
				else:
					eval_results[str(n_tasks)]['failures'] += 1
					eval_results[str(n_tasks)]['time'] += 7200 / n_reps
			
			else:
				raise InvalidEvaluationTypeError(evaluation)
			
		else:
			
			results['failures'] = 1
			results[metric] = 7200 if metric == 'time' else 0
			
			if evaluation == 'scale':
				if n_states not in eval_results:
					eval_results[str(n_states)] = dict()
					eval_results[str(n_states)][metric] = (7200 if metric == 'time' else 0) / n_reps
					eval_results[str(n_states)]['failures'] = 1
				else:
					eval_results[str(n_states)][metric] += (7200 if metric == 'time' else 0) / n_reps
					eval_results[str(n_states)]['failures'] += 1
			
			elif evaluation == 'goals':
				if n_tasks not in eval_results:
					eval_results[str(n_tasks)] = dict()
					eval_results[str(n_tasks)][metric] = (7200 if metric == 'time' else 0) / n_reps
					eval_results[str(n_tasks)]['failures'] = 1
				else:
					eval_results[str(n_tasks)][metric] += (7200 if metric == 'time' else 0) / n_reps
					eval_results[str(n_tasks)]['failures'] += 1
			
			else:
				raise InvalidEvaluationTypeError(evaluation)
	
	signal.alarm(0)
	# Dump iteration results to file
	print('Writing iteration results to file.')
	csv_file = data_dir / 'results' / ('evaluation_results_' + framework + '_' + evaluation + '_' + metric + '_' + world + '.csv')
	field_names = list(results.keys())

	if n_iteration < 1:
		write_iterations_results_csv(csv_file, results, 'w', field_names, state_goal, n_iteration)
	else:
		write_iterations_results_csv(csv_file, results, 'a', field_names, state_goal, n_iteration)
	

def world_evaluation(n_reps: int, beta: float, fail_prob: float, gamma: float, data_dir: Path, log_file: Path, evaluation: str, metric: str,
					 world: str, framework: str, world_idx: int, verbose: bool, state_goals_test: List[Tuple]) -> None:

	log_dir = log_file.parent.absolute()
	tested_environments_file = log_dir / ('evaluation_finished_' + framework + '_' + evaluation + '_' + metric + '.txt')
	tested_environments = []
	if tested_environments_file.exists():
		with open(tested_environments_file, 'r') as file:
			tested_environments = file.read().splitlines()
	else:
		tested_environments_file.touch(exist_ok=True)
	
	# If current world environment has already been tested then don't repeat
	if world in tested_environments:
		return
	
	sys.stdout = open(log_file, 'w')
	sys.stderr = open(log_file, 'a')
	print('Starting evaluation for world: ' + world)
	sys.stdout.flush()
	sys.stderr.flush()
	
	# Load mazeworld information
	with open(data_dir / 'configs' / (world + '.yaml')) as file:
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
	nT = len(tasks)
	
	# Create MDPs
	mdps = {}
	v_mdps = {}
	q_mdps = {}
	dists = []
	costs = []
	rewards = {}
	for i in range(len(tasks)):
		c = swmw.generate_rewards(tasks[i], X_w, A_w)
		costs += [c]
		rewards[tasks[i]] = c
		mdp = MDP(X_w, A_w, P_w, c, gamma, get_goal_states(X_w, tasks[i]), 'rewards', verbose)
		pol, q = mdp.policy_iteration()
		v = Utilities.v_from_q(q, pol)
		q_mdps[tasks[i]] = q
		v_mdps[tasks[i]] = v
		mdps['mdp' + str(i + 1)] = mdp
	dists = np.array(dists)
	
	# Verify if a savepoint exists to restart from
	savepoint_file = data_dir / 'results' / ('evaluation_' + framework + '_' + evaluation + '_' + metric + '_' + world + '.save')
	if savepoint_file.exists():
		print('Restarting evaluation. Loading savepoint.')
		eval_results, eval_begin = load_savepoint(savepoint_file)
	else:
		print('Starting evaluation from beginning.')
		eval_results = dict()
		eval_begin = 0
	
	iterator = tqdm(range(eval_begin, n_reps), desc='Evaluation Repetitions') if verbose else (range(eval_begin, n_reps))
	print('Starting evaluation iterations')
	for iteration in iterator:
		world_iteration(X_w, A_w, P_w, beta, evaluation, gamma, metric, nT, nX, n_reps, eval_results, task_states, tasks, framework, verbose,
						state_goals_test[iteration], q_mdps, v_mdps, mdps, dists, data_dir, iteration, world)
		if iteration % 10 == 0:
			store_savepoint(savepoint_file, eval_results, iteration)
			print('Updating evaluation savepoint.')
		if (iteration == 0) or ((iteration + 1) < 100 and (iteration + 1) % 10 == 0) or ((iteration + 1) > 100 and (iteration + 1) % 100 == 0):
			print('Reached iteration %d!! Completed %.2f%% of the repetitions\n' % ((iteration + 1), (iteration + 1) / n_reps * 100))
		sys.stdout.flush()
		sys.stderr.flush()
	
	# Dump world evaluation results to file
	print('Finished evaluation for world: %s\n' % world)
	csv_file = data_dir / 'results' / ('evaluation_results_' + framework + '_' + evaluation + '_' + metric + '.csv')
	results_keys = list(eval_results.keys())
	field_names = ['world_size'] + list(eval_results[results_keys[0]].keys())
		
	if world_idx == 1:
		write_full_results_csv(csv_file, metric, eval_results, 'w', field_names)
	else:
		write_full_results_csv(csv_file, metric, eval_results, 'a', field_names)
	
	# Update list of tested envrionments
	try:
		with open(tested_environments_file, 'a') as file:
			file.write(world + '\n')
			
	except IOError as e:
		print(colored("I/O error: " + str(e), color='red'))
	

def frameworks_evaluation(data_dir: Path, log_dir: Path, evaluation: str, n_reps: int, fail_prob: float, beta: float,
						  gamma: float, metric: str, framework: str, verbose: bool, test_keys: List[int]) -> None:
	
	# Get the correct environments to test and respective (initial state, goal) testing pairs
	if evaluation == 'scale':
		worlds = SCALABILITY_WORLDS
		worlds_keys = SCALABILITY_WORLDS.keys()
		with open(data_dir / 'configs' / 'scale_test.yaml') as file:
			state_goals = yaml.full_load(file)
	elif evaluation == 'goals':
		worlds = OBJECTS_WORLDS
		worlds_keys = OBJECTS_WORLDS.keys()
		with open(data_dir / 'configs' / 'goal_test.yaml') as file:
			state_goals = yaml.full_load(file)
	else:
		raise InvalidEvaluationTypeError(evaluation)
	
	# To increase efficiency parallelize evaluation, by running each world environment in a different subprocess
	procs = []
	for idx in worlds_keys:
		if idx in test_keys:
			log_file = log_dir / ('evaluation_log_' + framework + '_' + evaluation + '_' + metric + '_' + worlds[idx] + '.txt')
			p = Process(target=world_evaluation,
						args=(n_reps, beta, fail_prob, gamma, data_dir, log_file, evaluation, metric, worlds[idx],
							  framework, idx, verbose, state_goals[worlds[idx]]))
			p.start()
			procs.append(p)
	
	for p in procs:
		p.join()
	

def main():
	
	# Create argument parser
	parser = argparse.ArgumentParser(description='Legibility performance evaluation comparison between frameworks')
	parser.add_argument('--framework', dest='framework', type=str, required=True, choices=['policy', 'miura'],
						help='Framework to test')
	parser.add_argument('--evaluation', dest='evaluation_type', type=str, required=True, choices=['scale', 'goals'],
						help='Evaluation method to use.')
	parser.add_argument('--metric', dest='metric', type=str, required=True, choices=['miura', 'policy', 'time', 'all'],
						help='Metric to compare legibility performances.')
	parser.add_argument('--reps', dest='reps', type=int, required=True,
						help='Number of repetitions for the evaluation cycle for each framework.')
	parser.add_argument('--fail_prob', dest='fail_prob', type=float, required=True,
						help='Probability of movement failing.')
	parser.add_argument('--beta', dest='beta', type=float, required=True,
						help='Constant that guides how close to the optimal policy the legbility is.')
	parser.add_argument('--gamma', dest='gamma', type=float, required=True,
						help='Discount factor for the MDPs')
	parser.add_argument('--world-keys', dest='world_keys', type=int, nargs='+',
						help='List of world keys to test')
	parser.add_argument('--verbose', dest='verbose', action='store_true',
						help='Run MDPs methods with verbose printing')
	parser.add_argument('--no-verbose', dest='verbose', action='store_false',
						help='Run MDPs methods without verbose printing')
	# parser.add_argument('--name', dest='', type=, required=, choices=[],
	# 					help='')
	
	# Parsing input arguments
	args = parser.parse_args()
	framework = args.framework
	evaluation = args.evaluation_type
	metric = args.metric
	n_reps = args.reps
	fail_prob = args.fail_prob
	beta = args.beta
	gamma = args.gamma
	verbose = args.verbose
	world_keys = args.world_keys
	
	# Setup script output files and locations
	script_parent_dir = Path(__file__).parent.absolute().parent.absolute()
	data_dir = script_parent_dir / 'data'
	if not os.path.exists(script_parent_dir / 'logs'):
		os.mkdir(script_parent_dir / 'logs')
	log_dir = script_parent_dir / 'logs'
	log_file = log_dir / ('evaluation_log_' + framework + '_' + evaluation + '_' + metric + '.txt')
	sys.stdout = open(log_file, 'w+')
	sys.stderr = open(log_file, 'a')

	# By default test all world environments
	if world_keys is None:
		if evaluation == 'scale':
			world_keys = list(SCALABILITY_WORLDS.keys())
		elif evaluation == 'goals':
			world_keys = list(OBJECTS_WORLDS.keys())
		else:
			raise InvalidEvaluationTypeError(evaluation)

	try:
		print('Start time: ' + str(time.ctime()))
		frameworks_evaluation(data_dir, log_dir, evaluation, n_reps, fail_prob, beta, gamma, metric, framework, verbose, world_keys)
		print('End time: ' + str(time.ctime()))

	except InvalidEvaluationTypeError:
		print(colored('[ERROR] Invalid evaluation type, exiting program!', color='red'))
		raise

	except InvalidEvaluationMetricError:
		print(colored('[ERROR] Invalid metric, exiting program!', color='red'))
		raise
	
	sys.stdout.close()
	sys.stderr.close()


if __name__ == '__main__':
	main()
