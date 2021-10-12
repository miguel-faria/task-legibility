#! /usr/bin/env python
import sys
import os
import time
import signal

import numpy as np
import multiprocessing as mp
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
from pathlib import Path
from multiprocessing import Process

SCALABILITY_WORLDS = {1: '10x10_world_2', 2: '25x25_world',  3: '40x40_world', 4: '50x50_world',
					  5: '60x60_world', 6: '75x75_world',  7: '80x80_world', 8: '90x90_world'}

OBJECTS_WORLDS = {1: '25x25_world_g3', 2: '25x25_world_g4', 3: '25x25_world_g5', 4: '25x25_world_g6',
				  5: '25x25_world_g7', 6: '25x25_world_g8', 7: '25x25_world_g9', 8: '25x25_world_g10'}


class InvalidEvaluationTypeError(Exception):

	def __init__(self, eval_type, message="Evaluation type is not in list [scale, goals]"):
		self.eval_type = eval_type
		self.message = message
		super().__init__(self.message)

	def __str__(self):
		return f'{self.eval_type} -> {self.message}'


class InvalidEvaluationMetricError(Exception):

	def __init__(self, metric, message="Chosen evaluation metric is not among the possibilities: miura, policy or time"):
		self.metric = metric
		self.message = message
		super().__init__(self.message)

	def __str__(self):
		return f'{self.metric} -> {self.message}'
	

class InvalidFrameworkError(Exception):

	def __init__(self, metric, message="Chosen framework choice is not available. Available options are: policy or miura"):
		self.metric = metric
		self.message = message
		super().__init__(self.message)

	def __str__(self):
		return f'{self.metric} -> {self.message}'


class TimeoutException(Exception):

	def __init__(self, max_time, message="Could not finish evaluation in under 2 hours."):
		self.max_time = max_time
		self.message = message
		super().__init__(self.message)

	def __str__(self):
		return f'{self.max_time} -> {self.message}'


def signal_handler(signum, frame):
	raise TimeoutException(7200)


def write_csv(csv_file: Path, metric: str, results: Dict, access_type: str) -> None:
	try:
		if metric == 'all':
			results_keys = list(results.keys())
			fields = ['world_size'] + list(results[results_keys[0]].keys())
			with open(csv_file, access_type) as csvfile:
				writer = csv.DictWriter(csvfile, fieldnames=fields, delimiter=',', lineterminator='\n')
				if access_type != 'a':
					writer.writeheader()
				for key, val in sorted(results.items()):
					row = {'world_size': key}
					row.update(val)
					writer.writerow(row)
		else:
			fields = ['world_size'] + [metric]
			with open(csv_file, access_type) as csvfile:
				writer = csv.DictWriter(csvfile, fieldnames=fields, delimiter=',', lineterminator='\n')
				if access_type != 'a':
					writer.writeheader()
				for key, val in sorted(results.items()):
					row = {'world_size': key}
					row.update({metric: val})
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
					framework: str, verbose: bool, state_goal: Tuple, q_mdps: Dict, v_mdps: Dict, mdps: Dict, dists: np.ndarray) -> None:
	
	# Auxiliary methods to test the performance of each framework in obtaining a sequence of legible actions
	def policy_trajectory(task_mdp_w: LegibleTaskMDP, tasks: List[str], goal: str, x0: str) -> Tuple[np.ndarray, np.ndarray]:
		task_pol_w, _ = task_mdp_w.policy_iteration(tasks.index(goal))
		task_traj = task_mdp_w.trajectory(x0, task_pol_w)
		return task_traj
	
	def miura_trajectory(mdp: MDP, miura_mdp: MiuraLegibleMDP, x0: str, depth: int, n_its: int, beta: float, verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
		pol_w, _ = mdp.policy_iteration()
		miura_traj = miura_mdp.legible_trajectory(x0, pol_w, depth, n_its, beta, verbose)
		return miura_traj
	
	# Auxiliary methods to evaluate the legibility performance of each framework
	def policy_evaluation(trajectory: Tuple[np.ndarray, np.ndarray], task_idx: int, policy_mdp: LegibleTaskMDP) -> float:
		trajs = np.array([trajectory])
		return policy_mdp.trajectory_reward(trajs, task_idx)
	
	def miura_evaluation(trajectory: Tuple[np.ndarray, np.ndarray], task_idx: int, miura_mdp: MiuraLegibleMDP) -> float:
		trajs = np.array([trajectory])
		return miura_mdp.trajectory_reward(trajs, task_idx)
	
	# Choose current goal and obtain the corresponding goal states
	np.random.seed()
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
	try:
		if framework == 'policy':
			if metric == 'all':
				# Obtain sample trajectory with each framework
				policy_traj = policy_trajectory(policy_mdp, tasks, goal, x0)
				
				# Wrappers to run timeit on the trajectories for each framework
				policy_stmt = wrapper(policy_trajectory, policy_mdp, tasks, goal, x0)
				
				# Performance evaluation
				policy_m_performance = miura_evaluation(policy_traj, tasks.index(goal), miura_mdp)
				policy_p_performance = policy_evaluation(policy_traj, tasks.index(goal), policy_mdp)
				policy_t_performance = timeit.timeit(policy_stmt, number=1)
				
				if evaluation == 'scale':
					if n_states not in eval_results:
						eval_results[n_states] = dict()
						eval_results[n_states]['failures'] = 0
						eval_results[n_states]['policy'] = policy_p_performance / n_reps
						eval_results[n_states]['miura'] = policy_m_performance / n_reps
						eval_results[n_states]['time'] = policy_t_performance / n_reps
					else:
						eval_results[n_states]['policy'] += policy_p_performance / n_reps
						eval_results[n_states]['miura'] += policy_m_performance / n_reps
						eval_results[n_states]['time'] += policy_t_performance / n_reps
				
				elif evaluation == 'goals':
					if n_tasks not in eval_results:
						eval_results[n_tasks] = dict()
						eval_results[n_tasks]['failures'] = 0
						eval_results[n_tasks]['policy'] = policy_p_performance / n_reps
						eval_results[n_tasks]['miura'] = policy_m_performance / n_reps
						eval_results[n_tasks]['time'] = policy_t_performance / n_reps
					else:
						eval_results[n_tasks]['policy'] += policy_p_performance / n_reps
						eval_results[n_tasks]['miura'] += policy_m_performance / n_reps
						eval_results[n_tasks]['time'] += policy_t_performance / n_reps
				
				else:
					raise InvalidEvaluationTypeError(evaluation)
			
			else:
				if metric == 'miura':
					# Obtain sample trajectory with each framework
					policy_traj = policy_trajectory(policy_mdp, tasks, goal, x0)
					
					# Performance evaluation according to the metric in Miura et al. and storing
					policy_performance = miura_evaluation(policy_traj, tasks.index(goal), miura_mdp)
				
				elif metric == 'policy':
					# Obtain sample trajectory with each framework
					policy_traj = policy_trajectory(policy_mdp, tasks, goal, x0)
					
					# Performance evaluation according to the metric of policy legibility and storing
					policy_performance = policy_evaluation(policy_traj, tasks.index(goal), policy_mdp)
				
				elif metric == 'time':
					# Wrappers to run timeit on the trajectories for each framework
					policy_stmt = wrapper(policy_trajectory, policy_mdp, tasks, goal, x0)
					
					# Time each framework's performance and store it
					policy_performance = timeit.timeit(policy_stmt, number=1)
				
				else:
					raise InvalidEvaluationMetricError(metric)
				
				if evaluation == 'scale':
					if n_states not in eval_results:
						eval_results[n_states] = dict()
						eval_results[n_states]['failures'] = 0
						eval_results[n_states][metric] = policy_performance / n_reps
					else:
						eval_results[n_states][metric] += policy_performance / n_reps
				
				elif evaluation == 'goals':
					if n_tasks not in eval_results:
						eval_results[n_tasks] = dict()
						eval_results[n_tasks]['failures'] = 0
						eval_results[n_tasks][metric] = policy_performance / n_reps
					else:
						eval_results[n_tasks][metric] += policy_performance / n_reps
				
				else:
					raise InvalidEvaluationTypeError(evaluation)
		
		elif framework == 'miura':
			if metric == 'all':
				# Obtain sample trajectory with each framework
				miura_traj = miura_trajectory(mdps['mdp' + str(tasks.index(goal) + 1)], miura_mdp, x0, 10, 10000, beta, verbose)
				
				# Wrappers to run timeit on the trajectories for each framework
				miura_stmt = wrapper(miura_trajectory, mdps['mdp' + str(tasks.index(goal) + 1)], miura_mdp, x0, 10, 10000, beta, verbose)
				
				# Performance evaluation
				miura_m_performance = miura_evaluation(miura_traj, tasks.index(goal), miura_mdp)
				miura_p_performance = policy_evaluation(miura_traj, tasks.index(goal), policy_mdp)
				miura_t_performance = timeit.timeit(miura_stmt, number=1)
				
				if evaluation == 'scale':
					if n_states not in eval_results:
						eval_results[n_states] = dict()
						eval_results[n_states]['failures'] = 0
						eval_results[n_states]['policy'] = miura_p_performance / n_reps
						eval_results[n_states]['miura'] = miura_m_performance / n_reps
						eval_results[n_states]['time'] = miura_t_performance / n_reps
					else:
						eval_results[n_states]['policy'] += miura_p_performance / n_reps
						eval_results[n_states]['miura'] += miura_m_performance / n_reps
						eval_results[n_states]['time'] += miura_t_performance / n_reps
					
				elif evaluation == 'goals':
					if n_tasks not in eval_results:
						eval_results[n_tasks] = dict()
						eval_results[n_tasks]['failures'] = 0
						eval_results[n_tasks]['policy'] = miura_p_performance / n_reps
						eval_results[n_tasks]['miura'] = miura_m_performance / n_reps
						eval_results[n_tasks]['time'] = miura_t_performance / n_reps
					else:
						eval_results[n_tasks]['policy'] += miura_p_performance / n_reps
						eval_results[n_tasks]['miura'] += miura_m_performance / n_reps
						eval_results[n_tasks]['time'] += miura_t_performance / n_reps
				
				else:
					raise InvalidEvaluationTypeError(evaluation)
			
			else:
				if metric == 'miura':
					# Obtain sample trajectory with each framework
					miura_traj = miura_trajectory(mdps['mdp' + str(tasks.index(goal) + 1)], miura_mdp, x0, 10, 10000, beta, verbose)
					
					# Performance evaluation according to the metric in Miura et al. and storing
					miura_performance = miura_evaluation(miura_traj, tasks.index(goal), miura_mdp)
				
				elif metric == 'policy':
					# Obtain sample trajectory with each framework
					miura_traj = miura_trajectory(mdps['mdp' + str(tasks.index(goal) + 1)], miura_mdp, x0, 10, 10000, beta, verbose)
					
					# Performance evaluation according to the metric of policy legibility and storing
					miura_performance = policy_evaluation(miura_traj, tasks.index(goal), policy_mdp)
				
				elif metric == 'time':
					# Wrappers to run timeit on the trajectories for each framework
					miura_stmt = wrapper(miura_trajectory, mdps['mdp' + str(tasks.index(goal) + 1)], miura_mdp, x0, 10, 10000, beta, verbose)
					
					# Time each framework's performance and store it
					miura_performance = timeit.timeit(miura_stmt, number=1)
				
				else:
					raise InvalidEvaluationMetricError(metric)
				
				if evaluation == 'scale':
					if n_states not in eval_results:
						eval_results[n_states] = dict()
						eval_results[n_states]['failures'] = 0
						eval_results[n_states][metric] = miura_performance / n_reps
					else:
						eval_results[n_states][metric] += miura_performance / n_reps
				
				elif evaluation == 'goals':
					if n_tasks not in eval_results:
						eval_results[n_tasks] = dict()
						eval_results[n_tasks]['failures'] = 0
						eval_results[n_tasks][metric] = miura_performance / n_reps
					else:
						eval_results[n_tasks][metric] += miura_performance / n_reps
				
				else:
					raise InvalidEvaluationTypeError(evaluation)
		
		else:
			raise InvalidFrameworkError(framework)
	
	# If takes more than 2 horas to evaluate iteration
	except TimeoutException:
		
		if metric == 'all':
			if evaluation == 'scale':
				if n_states not in eval_results:
					eval_results[n_states] = dict()
					eval_results[n_states]['failures'] = 1
					eval_results[n_states]['policy'] = 0
					eval_results[n_states]['miura'] = 0
					eval_results[n_states]['time'] = 0
				else:
					eval_results[n_states]['failures'] += 1
			
			elif evaluation == 'goals':
				if n_tasks not in eval_results:
					eval_results[n_tasks] = dict()
					eval_results[n_tasks]['failures'] = 1
					eval_results[n_tasks]['policy'] = 0
					eval_results[n_tasks]['miura'] = 0
					eval_results[n_tasks]['time'] = 0
				else:
					eval_results[n_tasks]['failures'] += 1
			
			else:
				raise InvalidEvaluationTypeError(evaluation)
			
		else:
			if evaluation == 'scale':
				if n_states not in eval_results:
					eval_results[n_states] = dict()
					eval_results[n_states][metric] = 0
					eval_results[n_states]['failures'] = 1
				else:
					eval_results[n_states]['failures'] += 1
			
			elif evaluation == 'goals':
				if n_tasks not in eval_results:
					eval_results[n_tasks] = dict()
					eval_results[n_tasks][metric] = 0
					eval_results[n_tasks]['failures'] = 1
				else:
					eval_results[n_tasks]['failures'] += 1
			
			else:
				raise InvalidEvaluationTypeError(evaluation)
	

def world_evaluation(n_reps: int, beta: float, fail_prob: float, gamma: float, data_dir: Path, log_file: Path, evaluation: str, metric: str,
					 world: str, framework: str, world_idx: int, verbose: bool, state_goals_test: List[Tuple]) -> None:

	sys.stdout = open(log_file, 'w')
	sys.stderr = open(log_file, 'a')
	print('Starting evaluation for world: ' + world)
	sys.stdout.flush()
	sys.stderr.flush()
	# pool = mp.Pool(processes=mp.cpu_count())
	
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
	
	eval_results = dict()
	# manager = mp.Manager()
	# eval_results = manager.dict()
	iterator = tqdm(range(n_reps), desc='Evaluation Repetitions') if verbose else (range(n_reps))
	for iteration in iterator:
		# pool.apply_async(world_iteration, args=(X_w, A_w, P_w, beta, evaluation, gamma, metric, nT, nX, n_reps,
		# 										eval_results, swmw, task_states, tasks, framework, q_mdps, v_mdps, mdps, dists))
		world_iteration(X_w, A_w, P_w, beta, evaluation, gamma, metric, nT, nX, n_reps, eval_results,
						task_states, tasks, framework, verbose, state_goals_test[iteration], q_mdps, v_mdps, mdps, dists)
		if (iteration == 0) or ((iteration + 1) < 100 and (iteration + 1) % 10 == 0) or ((iteration + 1) > 100 and (iteration + 1) % 100 == 0):
			print('Reached iteration %d!! Completed %.2f%% of the repetitions\n' % ((iteration + 1), (iteration + 1) / n_reps * 100))
		sys.stdout.flush()
		sys.stderr.flush()
	
	print('Finished evaluation for world: %s\n' % world)
	csv_file = data_dir / 'results' / ('evaluation_results_' + framework + '_' + evaluation + '_' + metric + '.csv')
	if world_idx < 1:
		write_csv(csv_file, metric, eval_results, 'w')
	else:
		write_csv(csv_file, metric, eval_results, 'a')
	
	# pool.close()
	# pool.join()


def frameworks_evaluation(data_dir: Path, log_dir: Path, evaluation: str, n_reps: int, fail_prob: float, beta: float,
						  gamma: float, metric: str, framework: str, verbose: bool) -> None:
	
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
	
	world_idx = 0
	procs = []
	for idx in worlds_keys:
		log_file = log_dir / ('evaluation_log_' + framework + '_' + evaluation + '_' + metric + '_' + worlds[idx] + '.txt')
		p = Process(target=world_evaluation,
					args=(n_reps, beta, fail_prob, gamma, data_dir, log_file, evaluation, metric, worlds[idx],
						  framework, world_idx, verbose, state_goals[worlds[idx]]))
		world_idx += 1
		p.start()
		procs.append(p)
		# world_evaluation(n_reps, beta, fail_prob, gamma, data_dir, log_file, evaluation, metric, worlds[idx], framework, world_idx)
	
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
	parser.add_argument('--verbose', dest='verbose', action='store_true',
						help='Discount factor for the MDPs')
	parser.add_argument('--no-verbose', dest='verbose', action='store_false',
						help='Discount factor for the MDPs')
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
	
	# Setup script output files and locations
	script_parent_dir = Path(__file__).parent.parent.absolute()
	data_dir = script_parent_dir / 'data'
	if not os.path.exists(script_parent_dir / 'logs'):
		os.mkdir(script_parent_dir / 'logs')
	log_dir = script_parent_dir / 'logs'
	log_file = log_dir / ('evaluation_log_' + framework + '_' + evaluation + '_' + metric + '.txt')
	sys.stdout = open(log_file, 'w+')
	sys.stderr = open(log_file, 'a')

	try:
		print('Start time: ' + str(time.ctime()))
		frameworks_evaluation(data_dir, log_dir, evaluation, n_reps, fail_prob, beta, gamma, metric, framework, verbose)
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
