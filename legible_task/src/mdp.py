#! /usr/bin/env python

from __future__ import annotations
import numpy as np
import time
import math
from abc import ABC
from tqdm import tqdm
from termcolor import colored
from typing import List, Dict, Tuple, Callable
from src.mdpworld import MDPWorld


class Utilities(object):

	@staticmethod
	def state_tuple_to_str(state: tuple):
		return ', '.join(' '.join(str(x) for x in elem) for elem in state)

	@staticmethod
	def state_str_to_tuple(state_str: str):
		state_tuple = []
		state_split = state_str.split(', ')
		for elem in state_split:
			elem_split = elem.split(' ')
			elem_tuple = []
			for elem_2 in elem_split:
				try:
					elem_tuple += [int(elem_2)]
				except ValueError:
					elem_tuple += [elem_2]
			state_tuple += [tuple(elem_tuple)]
		
		return tuple(state_tuple)

	@staticmethod
	def q_from_pol(mdp: MDP, pol: np.ndarray, task_idx: int = None) -> np.ndarray:
		
		X = mdp.states
		A = mdp.actions
		P = mdp.transitions_prob
		if task_idx is not None:
			c = mdp.costs[task_idx]
		else:
			c = mdp.costs
		gamma = mdp.gamma
		nX, nA = len(X), len(A)
		
		q = np.zeros((nX, nA))
		J = mdp.evaluate_pol(pol, task_idx)
		
		for act in range(nA):
			q[:, act, None] = c[:, act, None] + gamma * P[A[act]].dot(J)
		
		return q
	
	@staticmethod
	def q_from_v(v: np.ndarray, mdp: MDP) -> np.ndarray:
		
		A = mdp.actions
		P = mdp.transitions_prob
		c = mdp.costs
		gamma = mdp.gamma
		
		nX = len(mdp.states)
		nA = len(A)
		q = np.zeros((nX, nA))
		
		for a_idx in range(nA):
			q[:, a_idx, None] = c[:, a_idx, None] + gamma * P[A[a_idx]].dot(v)
		
		return q

	@staticmethod
	def v_from_pol(mdp: MDP, pol: np.ndarray, feedback_type: str, task_idx: int = None) -> np.ndarray:
		if feedback_type == 'costs':
			return np.min(Utilities.q_from_pol(mdp, pol, task_idx), axis=1)[:, None]
		elif feedback_type == 'rewards':
			return np.max(Utilities.q_from_pol(mdp, pol, task_idx), axis=1)[:, None]
		else:
			return np.zeros((len(mdp.states), 1))

	@staticmethod
	def v_from_q(q: np.ndarray, feedback_type: str) -> np.ndarray:
		if feedback_type == 'costs':
			return np.min(q, axis=1)[:, None]
		elif feedback_type == 'rewards':
			return np.max(q, axis=1)[:, None]
		else:
			return np.zeros((q.shape[0], 1))
	
	@staticmethod
	def Ppi(transitions: Dict[str, np.ndarray], actions: np.ndarray, pol: np.ndarray) -> np.ndarray:

		nA = len(actions)
		ppi = pol[:, 0, None] * transitions[actions[0]]
		for i in range(1, nA):
			ppi += pol[:, i, None] * transitions[actions[i]]
			
		return ppi

	@staticmethod
	def Ppi_stack(transitions: Dict[str, np.ndarray], actions: np.ndarray, pol: np.ndarray) -> np.ndarray:
		
		nX = len(transitions[actions[0]])
		nA = len(actions)
		ppi = np.zeros((nX*nA, nX*nA))
		
		for a in range(nA):
			for a2 in range(nA):
				ppi[a*nX:(a+1)*nX, a2*nX:(a2+1)*nX] = transitions[actions[a]] * pol[:, a2, None]
		
		return ppi

	@staticmethod
	def Q_func_grad(transitions: Dict[str, np.ndarray], actions: np.ndarray, pol: np.ndarray, q_pi: np.ndarray, gamma: float) -> np.ndarray:
		
		nX, nA = q_pi.shape
		q_stack = q_pi.flatten('F')
		pi_stack = pol.flatten('F')
		ppi_stack = Utilities.Ppi_stack(transitions, actions, pol)
		
		q_grad = gamma * np.linalg.inv(np.eye(nX*nA) - gamma * ppi_stack).dot(ppi_stack)*np.diag(q_stack/pi_stack)
		
		return np.diagonal(q_grad).reshape((nX, nA), order='F')
		
	@staticmethod
	def softmax(param: np.ndarray, temp: float) -> np.ndarray:
	
		tmp = np.exp((param - np.max(param, axis=1)[:, None]) / temp)
		return tmp/tmp.sum(axis=1)[:, None]
	
	@staticmethod
	def softmax_grad(param: np.ndarray, temp: float) -> np.ndarray:
		
		softmax_pol = Utilities.softmax(param, temp)
		nX, nA = softmax_pol.shape
		sofmax_grad = np.zeros((nX*nA, nX*nA))
		
		for x in range(nX):
			for a in range(nA):
				for a2 in range(nA):
					sofmax_grad[x*a, x*a2] = softmax_pol[x, a] * ((1 if a == a2 else 0) - softmax_pol[x, a2])
			
		return np.diagonal(sofmax_grad).reshape((nX, nA), order='F')


class MDP(object):
	
	def __init__(self, x: np.ndarray, a: np.ndarray, p: Dict[str, np.ndarray], c: np.ndarray, gamma: float, goal_states: List[int], feedback_type: str,
				 verbose: bool):
		self._mdp = (x, a, p, c, gamma)
		self._goal_states = goal_states
		self._verbose = verbose
		if feedback_type.lower().find('cost') != -1 or feedback_type.lower().find('reward') != -1:
			self._feedback_type = feedback_type
		else:
			print('Invalid feedback type, defaulting to use costs.')
			self._feedback_type = 'cost'
	
	@property
	def states(self) -> np.ndarray:
		return self._mdp[0]
	
	@property
	def actions(self) -> np.ndarray:
		return self._mdp[1]
	
	@property
	def transitions_prob(self) -> Dict[str, np.ndarray]:
		return self._mdp[2]
	
	@property
	def costs(self) -> np.ndarray:
		return self._mdp[3]
	
	@property
	def gamma(self) -> float:
		return self._mdp[4]
	
	@property
	def goals(self) -> List[int]:
		return self._goal_states
	
	@property
	def mdp(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray, float]:
		return self._mdp
	
	@property
	def verbose(self) -> bool:
		return self._verbose
	
	@mdp.setter
	def mdp(self, mdp: Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray, float]):
		self._mdp = mdp
	
	@goals.setter
	def goals(self, goals: List[str]):
		self._goal_states = goals
	
	@verbose.setter
	def verbose(self, verbose: bool):
		self._verbose = verbose
	
	def update_states(self, X_new: np.ndarray):
		
		_, A, P, c, gamma = self._mdp
		self._mdp = (X_new, A, P, c, gamma)
	
	def update_actions(self, A_new: np.ndarray):
		X, _, P, c, gamma = self._mdp
		self._mdp = (X, A_new, P, c, gamma)
	
	def update_transitions(self, P_new: Dict[str, np.ndarray]):
		X, A, _, c, gamma = self._mdp
		self._mdp = (X, A, P_new, c, gamma)
	
	def update_costs_rewards(self, c_new: np.ndarray, feedback_type: str):
		X, A, P, _, gamma = self._mdp
		self._mdp = (X, A, P, c_new, gamma)
		if feedback_type.lower().find('cost') != -1 or feedback_type.lower().find('reward') != -1:
			self._feedback_type = feedback_type
		else:
			print('Invalid feedback type, defaulting to use costs.')
			self._feedback_type = 'cost'
	
	def update_gamma(self, gamma_new: float):
		X, A, P, c, _ = self._mdp
		self._mdp = (X, A, P, c, gamma_new)
		
	def get_possible_states(self, q: np.ndarray) -> np.ndarray:
		nonzerostates = np.nonzero(q.sum(axis=1))[0]
		possible_states = [np.delete(nonzerostates, np.argwhere(nonzerostates == g)) for g in self._goal_states][0]
		return possible_states
	
	def evaluate_pol(self, pol: np.ndarray, task_idx: int = None) -> np.ndarray:
		X = self._mdp[0]
		A = self._mdp[1]
		P = self._mdp[2]
		if task_idx is None:
			c = self._mdp[3]
		else:
			c = self._mdp[3][task_idx]
		gamma = self._mdp[4]
		
		nS = len(X)
		
		# Cost and Probs averaged by policy
		cpi = (pol * c).sum(axis=1)
		ppi = Utilities.Ppi(P, A, pol)
		
		# J = (I - gamma*P)^-1 * c
		J = np.linalg.inv(np.eye(nS) - gamma * ppi).dot(cpi)
		
		return J[:, None]
	
	def value_iteration(self, task_idx: int = None) -> np.ndarray:
		X = self._mdp[0]
		A = self._mdp[1]
		P = self._mdp[2]
		if task_idx is not None:
			c = self._mdp[3][task_idx]
		else:
			c = self._mdp[3]
		gamma = self._mdp[4]
		
		nS = len(X)
		nA = len(A)
		
		J = np.zeros(nS)
		err = 1
		i = 0
		
		while err > 1e-8:
			Q = []
			for act in range(nA):
				Q += [c[:, act] + gamma * P[A[act]].dot(J)]
			
			if self._feedback_type.find('cost') != -1:
				Jnew = np.min(Q, axis=0)
			else:
				Jnew = np.max(Q, axis=0)
			err = np.linalg.norm(J - Jnew)
			J = Jnew
			
			i += 1
		
		return J[:, None]
	
	def policy_iteration(self, task_idx: int = None) -> (np.ndarray, np.ndarray):
		X = self._mdp[0]
		A = self._mdp[1]
		P = self._mdp[2]
		if task_idx is not None:
			c = self._mdp[3][task_idx]
		else:
			c = self._mdp[3]
		gamma = self._mdp[4]
		
		nS = len(X)
		nA = len(A)
		
		# Initialize pol
		pol = np.ones((nS, nA)) / nA
		
		# Initialize Q
		Q = np.zeros((nS, nA))
		
		quit = False
		i = 0
		
		while not quit:
			if self._verbose:
				print('Iteration %d' % (i + 1), end='\r')
			
			J = self.evaluate_pol(pol, task_idx)
			
			for act in range(nA):
				Q[:, act, None] = c[:, act, None] + gamma * P[A[act]].dot(J)
			
			if self._feedback_type.find('cost') != -1:
				Qbest = Q.min(axis=1, keepdims=True)
			else:
				Qbest = Q.max(axis=1, keepdims=True)
			polnew = np.isclose(Q, Qbest, atol=1e-10, rtol=1e-10).astype(int)
			polnew = polnew / polnew.sum(axis=1, keepdims=True)
			
			quit = (pol == polnew).all( )
			pol = polnew
			i += 1
		
		if self._verbose:
			print('N. iterations: ', i)
		
		return pol, Q
	
	def trajectory_len(self, x0: str, pol: np.ndarray, traj_len: int, rng_gen: np.random.Generator) -> (np.ndarray, np.ndarray):
		X = self._mdp[0]
		A = self._mdp[1]
		P = self._mdp[2]
		
		nX = len(X)
		nA = len(A)
		
		traj = [x0]
		actions = []
		x = list(X).index(x0)
		
		for _ in range(traj_len):
			a = rng_gen.choice(nA, p=pol[x, :])
			x = rng_gen.choice(nX, p=P[A[a]][x, :])
			
			traj += [X[x]]
			actions += [A[a]]
		
		actions += [A[rng_gen.choice(nA, p=pol[x, :])]]
		return np.array(traj), np.array(actions)
	
	def trajectory(self, x0: str, pol: np.ndarray, rng_gen: np.random.Generator) -> (np.ndarray, np.ndarray):
		X = self._mdp[0]
		A = self._mdp[1]
		P = self._mdp[2]
		
		nX = len(X)
		nA = len(A)
		
		traj = [x0]
		actions = []
		x = list(X).index(x0)
		stop = False
		i = 0
		
		while not stop:
			a = rng_gen.choice(nA, p=pol[x, :])
			x = rng_gen.choice(nX, p=P[A[a]][x, :])
			
			traj += [X[x]]
			actions += [A[a]]

			stop = (x in self._goal_states or i > 500)
			if stop:
				actions += [A[rng_gen.choice(nA, p=pol[x, :])]]

			i += 1
		
		return np.array(traj), np.array(actions)
	
	def all_trajectories(self, x0: str, pol: np.ndarray, rng_gen: np.random.Generator) -> (np.ndarray, np.ndarray):
		X = self._mdp[0]
		A = self._mdp[1]
		P = self._mdp[2]
		
		nX = len(X)
		nA = len(A)
		
		i = 0
		trajs = []
		acts = []
		started_trajs = [[[x0], []]]
		stop = False

		while not stop:

			traj = started_trajs[i][0]
			a_traj = started_trajs[i][1]
			x = list(X).index(traj[-1])
			stop_inner = False
			it = 0
			add_traj = False

			while not stop_inner:
				if x in self._goal_states:
					a_traj += [A[rng_gen.choice(nA, p=pol[x, :])]]
					add_traj = True
					break
				
				else:
					pol_act = np.nonzero(pol[x, :])[0]
					if len(pol_act) > 1:
						for j in range(1, len(pol_act)):
							if len(np.nonzero(P[A[pol_act[j]]][x, :])) > 1:
								x_tmp = rng_gen.choice(nX, p=P[A[pol_act[j]]][x, :])
								while x_tmp == x:
									x_tmp = rng_gen.choice(nX, p=P[A[pol_act[j]]][x, :])
							else:
								x_tmp = rng_gen.choice(nX, p=P[A[pol_act[j]]][x, :])
							tmp_traj = [list(traj) + [X[x_tmp]], list(a_traj) + [A[pol_act[j]]]]
							if tmp_traj not in started_trajs:
								started_trajs += [tmp_traj]
					
					a = pol_act[0]
					x = rng_gen.choice(nX, p=P[A[a]][x, :])
					
					if X[x] != traj[-1]:
						traj += [X[x]]
						a_traj += [A[a]]
						
						stop_inner = (x in self._goal_states)
						if stop_inner:
							a_traj += [A[rng_gen.choice(nA, p=pol[x, :])]]
						
						add_traj = True
					
					else:
						if it > 500:
							stop_inner = True
					
					it += 1
			
			i += 1
			stop = (i >= len(started_trajs) or i > 500)
			if add_traj:
				trajs += [np.array(traj)]
				acts += [np.array(a_traj)]
		
		return np.array(trajs, dtype=object), np.array(acts, dtype=object)
	
	def trajectory_reward(self, trajs: np.ndarray, task_idx: int = None) -> float:
		r_avg = 0
		X = list(self._mdp[0])
		A = list(self._mdp[1])
		if task_idx is not None:
			c = self._mdp[3][task_idx]
		else:
			c = self._mdp[3]
		gamma = self._mdp[4]
		n_trajs = len(trajs)
		t_idx = 1
		
		for traj in trajs:
			states = traj[0]
			actions = traj[1]
			r_traj = 0
			g = 1
			
			for idx in range(len(actions)):
				x = states[idx]
				a = actions[idx]
				r_traj += g * c[X.index(x), A.index(a)]
				g *= gamma
			
			r_avg += r_traj / n_trajs
			if self._verbose:
				print('MDP trajectory reward! Trajectory nr.%d. %.2f%% done' % (t_idx, t_idx / n_trajs * 100), end='\r')
				t_idx += 1
		
		if self._verbose:
			print('MDP trajectory reward finished!', end='\n')
		
		return r_avg

	def avg_dist(self, x0: str, pol: np.ndarray, rng_gen: np.random.Generator) -> int:

		trajs, _ = self.all_trajectories(x0, pol, rng_gen)
		dist = 0
		n_trajs = len(trajs)
		for traj in trajs:
			dist += len(traj) / n_trajs

		return math.ceil(dist)

	def policy_dist(self, pol: np.ndarray, rng_gen: np.random.Generator, task_idx: int = None) -> np.ndarray:

		# dists = np.zeros(len(self.states))
		dists = np.ones(len(self.states)) * 1000
		q_pol = Utilities.q_from_pol(self, pol, task_idx)
		possible_states = list(self.get_possible_states(q_pol))
		for x in possible_states:
			dists[x] = self.avg_dist(self.states[x], pol, rng_gen)

		return dists


class LegibleTaskMDP(MDP):
	
	def __init__(self, x: np.ndarray, a: np.ndarray, p: Dict[str, np.ndarray], gamma: float, verbose: bool, task: str, task_states: List[Tuple[int, int, str]],
				 tasks: List[str], beta: float, goal_states: List[int], sign: int, leg_func: str, q_mdps: Dict[str, np.ndarray], v_mdps: Dict[str, np.ndarray],
				 dists: np.ndarray):
		self._legible_functions = {'leg_optimal': self.optimal_legible_cost, 'leg_weight': self.legible_cost}
		self._task = task
		self._tasks = tasks
		self._task_states = {}
		for task_state in task_states:
			self._task_states[task_state[2]] = tuple([task_state[0], task_state[1]])
		self._tasks_q = q_mdps
		self._v_mdps = v_mdps
		self._beta = beta
		nX = len(x)
		nA = len(a)
		nT = len(tasks)
		c = np.zeros((nT, nX, nA))
		
		super().__init__(x, a, p, c, gamma, goal_states, 'rewards', verbose)
		if leg_func in list(self._legible_functions.keys()):
			for t in range(nT):
				for i in range(nX):
					for j in range(nA):
						c[t, i, j] = self._legible_functions[leg_func](i, j, t, sign, dists)
			
			self._mdp = (x, a, p, c, gamma)
		
		else:
			print(colored('Invalid legibility function. Exiting without computing cost function.',
						  'red'))
			return
	
	def update_cost_function(self, legible_func: str, sign: int, dists: np.ndarray):
		mdp = self.mdp
		x = mdp[0]
		a = mdp[1]
		p = mdp[2]
		gamma = mdp[4]
		nX = len(x)
		nA = len(a)
		nT = len(self._tasks)
		c = np.zeros((nT, nX, nA))
		
		if legible_func in list(self._legible_functions.keys( )):
			for t in range(nT):
				for i in range(nX):
					for j in range(nA):
						c[t, i, j] = self._legible_functions[legible_func](i, j, t, sign, dists)
			
			self._mdp = (x, a, p, c, gamma)
		
		else:
			print(colored('Invalid legibility function. Exiting without computing cost function.',
						  'red'))
			return
	
	def optimal_legible_cost(self, x: int, a: int, t: int, sign: int, dist: np.ndarray) -> float:
		task_cost = np.exp(sign * self._beta * self._tasks_q[self._tasks[t]][x, a])
		tasks_sum = task_cost
		for task in self._tasks:
			if task != self._tasks[t]:
				tasks_sum += np.exp(sign * self._beta * self._tasks_q[task][x, a])
		
		return task_cost / tasks_sum

	def legible_cost(self,  x: int, a: int, t: int, sign: int, dist: np.ndarray) -> float:
		task_cost = np.exp(sign * self._beta * self._tasks_q[self._tasks[t]][x, a]) / (1 / dist[t, x])
		tasks_sum = task_cost
		for task in self._tasks:
			if task != self._tasks[t]:
				tasks_sum += np.exp(sign * self._beta * self._tasks_q[task][x, a]) / (1 / dist[self._tasks.index(task), x])

		return task_cost / tasks_sum

	def pol_legibility(self, tasks_q_pi: Dict[str, np.ndarray], eta: float) -> float:
		
		nX, nA = tasks_q_pi[self._task].shape
		
		task_prob = (np.exp(eta * (tasks_q_pi[self._task] - self._v_mdps[self._task][:, None])) / (nX * nA)).sum()
		task_prob_sum = []
		
		for task in self._tasks:
			task_prob_sum += [(np.exp(eta * (tasks_q_pi[task] - self._v_mdps[task][:, None])) / (nX * nA)).sum()]
			
		task_prob_sum = np.array(task_prob_sum)
		return task_prob / task_prob_sum.sum()
	
	def legibility_optimization(self, task_rewards: Dict[str, np.ndarray], learn_rate: float, eta: float, improv_thresh: float,
								softmax_temp: float, init_params: np.ndarray, best_action: bool = False) -> np.ndarray:
		
		def get_policy(param: np.ndarray, temperature: float) -> np.ndarray:
			return Utilities.softmax(param, temperature)

		def compute_q_pi(rewards: Dict[str, np.ndarray], pi: np.ndarray) -> Dict[str, np.ndarray]:
			
			q_pis = {}
			nX, nA = pi.shape
			
			for task in self._tasks:
				
				Q = np.zeros((nX, nA))
				J = self.evaluate_pol(pi, self._tasks.index(task))
				
				for a in range(nA):
					Q[:, a, None] = rewards[task][:, a, None] + self.gamma * self.transitions_prob[self.actions[a]].dot(J)
				
				q_pis[task] = Q
			
			return q_pis
		
		def theta_gradient(tasks_q_pi: Dict[str, np.ndarray], pi: np.ndarray, params: np.ndarray, temp: float) -> np.ndarray:
			
			goal_q_grad = Utilities.Q_func_grad(self.transitions_prob, self.actions, pi, tasks_q_pi[self._task], self.gamma)
			gradient = np.zeros(tasks_q_pi[self._task].shape)
			softmax_grad = Utilities.softmax_grad(params, temp)
			
			for task in self._tasks:
				task_q_grad = Utilities.Q_func_grad(self.transitions_prob, self.actions, pi, tasks_q_pi[task], self.gamma)
				gradient += (eta * (goal_q_grad - task_q_grad)) * np.exp(eta * (tasks_q_pi[self._task] + self._v_mdps[task][:, None] -
																				tasks_q_pi[task] - self._v_mdps[self._task][:, None]))

			return gradient * softmax_grad
		
		stop = False
		theta = init_params
		pol = get_policy(theta, softmax_temp)
		tasks_q = compute_q_pi(task_rewards, pol)
		legibility = self.pol_legibility(tasks_q, eta)
		it = 0
		
		while not stop:
			
			if self._verbose:
				print('Iteration: %d\t Legibility: %.8f' % (it + 1, legibility))
			# improve policy
			theta_grad = theta_gradient(tasks_q, pol, theta, softmax_temp)
			# print(theta_grad)
			theta += learn_rate * theta_grad
			# print(theta)
			pol_new = get_policy(theta, softmax_temp)
			
			# eval policy
			tasks_q = compute_q_pi(task_rewards, pol_new)
			new_legibility = self.pol_legibility(tasks_q, eta)
			improv = new_legibility - legibility
			stop = (improv < improv_thresh)
			it += 1
			if not stop:
				legibility = new_legibility
				pol = pol_new
		
		if self._verbose:
			print('Took %d iterations\t Legibility: %.8f' % (it + 1, legibility))
		
		if best_action:
			pol_best = pol.max(axis=1, keepdims=True)
			pol = np.isclose(pol, pol_best, atol=1e-10, rtol=1e-10).astype(int)
			pol = pol / pol.sum(axis=1, keepdims=True)
		
		return pol


class MiuraLegibleMDP(MDP):
	
	class MCTSMDPNode(ABC):
		
		def __init__(self, num_actions: int, node_state: int, mdp: MDP, parent_node=None, terminal: bool = False):
			self._num_actions = num_actions
			self._q = np.zeros(self._num_actions)
			self._n = np.ones(self._num_actions)
			self._state = node_state
			self._parent = parent_node
			self._children = dict()
			self._terminal = terminal
			self._mdp = mdp
		
		def expand(self, action: str, depth: int, discount: float, goal_states: List[int], pol: np.ndarray) -> (float, List[Tuple]):
			act_idx = list(self._mdp.actions).index(action)
			next_node, reward = self.simulate_action(action, goal_states)
			
			if self._terminal:
				action_reward = reward
				history = [(self._mdp.states[self._state],)]
			
			else:
				future_reward, future_choices = next_node.rollout(depth - 1, discount, goal_states, pol)
				action_reward = reward + discount * future_reward
				history = [(self._mdp.states[self._state], action)] + future_choices
			
			self._n[act_idx] += 1
			self._q[act_idx] += (action_reward - self._q[act_idx]) / self._n[act_idx]
			
			return action_reward, history
		
		def rollout(self, depth: int, discount: float, goal_states: List[int], pol: np.ndarray) -> (float, List[Tuple]):
			
			rng_gen = np.random.default_rng(int(time.time()))
			A = self._mdp.actions
			if depth == 0:
				return 0.0, [(self._mdp.states[self._state],)]
			
			action = A[rng_gen.choice(range(self._num_actions), p=pol[self._state, :])]
			next_node, reward = self.simulate_action(action, goal_states)
			
			if self._terminal:
				return reward, [(self._mdp.states[self._state],)]
			else:
				future_reward, future_choices = next_node.rollout(depth - 1, discount, goal_states, pol)
				return reward + discount * future_reward, [(self._mdp.states[self._state], action)] + future_choices
		
		def simulate_action(self, action: str, goal_states: List[int]) -> (int, float):
			
			rng_gen = np.random.default_rng(int(time.time()))
			nX = len(self._mdp.states)
			A = list(self._mdp.actions)
			act_idx = A.index(action)
			P = self._mdp.transitions_prob
			c = self._mdp.costs
			
			next_state = rng_gen.choice(nX, p=P[action][self._state, :])
			reward = c[next_state, act_idx]
			
			terminal_state = next_state in goal_states
			next_node = MiuraLegibleMDP.MCTSMDPNode(self._num_actions, next_state, self._mdp, self, terminal_state)
			return next_node, reward
		
		def simulate(self, depth: int, exploration: float, discount_factor: float, visited_nodes: List['MCTSMDPNode'],
					 goal_states: List[int], pol: np.ndarray) -> float:
			A = self._mdp.actions
			if depth == 0:
				return 0.0
			
			if self not in visited_nodes:
				visited_nodes.append(self)
				for act_idx in range(self._num_actions):
					self.expand(A[act_idx], depth, discount_factor, goal_states, pol)
				node = self
			else:
				node = visited_nodes[visited_nodes.index(self)]
			
			act_idx = node.ucb_alternative().argmax()
			q, history = self.expand(A[act_idx], depth, discount_factor, goal_states, pol)
			
			return q
		
		def ucb(self, c: float) -> np.ndarray:
			return self._q + c * np.sqrt(2 * np.log(self._n.sum()) / self._n)
		
		def ucb_alternative(self) -> np.ndarray:
			return self._q + self._q * np.sqrt(2 * np.log(self._n.sum()) / self._n)
		
		def uct(self, goal_states: List[int], max_iterations: int, max_depth: int, exploration: float, pol: np.ndarray, verbose: bool) -> int:
			visited_nodes = []
			iterator = tqdm(range(max_iterations)) if verbose else range(max_iterations)
			for _ in iterator:
				self.simulate(max_depth, exploration, self._mdp.mdp[4], visited_nodes, goal_states, pol)
			
			return self._q.argmin()
	
	class MiuraMDPMCTSNode(MCTSMDPNode):
		
		def __init__(self, num_actions: int, node_state: int, mdp: 'MiuraLegibleMDP', belief: np.ndarray,
					 parent_node=None, terminal: bool = False):
			super(MiuraLegibleMDP.MiuraMDPMCTSNode, self).__init__(num_actions, node_state, mdp, parent_node, terminal)
			self._belief = belief
			self._objectives = mdp.tasks
			self._objective = mdp.goal
		
		def simulate_action(self, action: str, goal_states: List[int], rng_gen: np.random.Generator) -> (int, float):
			nX = len(self._mdp.states)
			P = self._mdp.transitions_prob
			
			next_state = rng_gen.choice(nX, p=P[action][self._state, :])
			belief = self._mdp.update_belief(self._state, action, next_state, self._belief)
			reward = self._mdp.belief_reward(belief[self._objectives.index(self._objective)])
			
			terminal_state = next_state in goal_states
			next_node = MiuraLegibleMDP.MiuraMDPMCTSNode(self._num_actions, next_state, self._mdp, belief, self, terminal_state)
			return next_node, reward
	
	def __init__(self, x: np.ndarray, a: List[str], p: Dict[str, np.ndarray], gamma: float, verbose: bool, task: str, tasks: List[str], beta: float,
				 goal_states: List[int], q_mdps: Dict[str, np.ndarray]):
		
		nX = len(x)
		nA = len(a)
		nT = len(tasks)
		c = np.zeros((nT, nX, nA))
		super().__init__(x, a, p, c, gamma, goal_states, 'rewards', verbose)
		self._q_mdps = q_mdps
		self._beta = beta
		self._goal = task
		self._tasks = tasks
	
	@property
	def tasks(self):
		return self._tasks
	
	@property
	def goal(self):
		return self._goal
	
	def belief_reward(self, belief: float) -> float:
		return -np.sqrt(1**2 - belief**2)
	
	def update_belief(self, x: int, a: str, x1: int, curr_belief: np.ndarray) -> np.ndarray:
		
		A = list(self._mdp[1])
		P = self._mdp[2]
		a_idx = A.index(a)
		new_belief = np.zeros(len(self._tasks))
		for task in self._tasks:
			task_idx = self._tasks.index(task)
			new_belief[task_idx] = P[a][x, x1] * np.exp(self._beta * self._q_mdps[task][x, a_idx]) * curr_belief[task_idx]

		return new_belief / new_belief.sum()

	def legible_trajectory(self, x0: str, pol: np.ndarray, depth: int, n_its: int, beta: float,
						   verbose: bool, rng_gen: np.random.Generator) -> (np.ndarray, np.ndarray):
		X = self._mdp[0]
		A = self._mdp[1]
		P = self._mdp[2]
		
		nX = len(X)
		nA = len(A)
		nT = len(self._tasks)
		
		traj = [x0]
		actions = []
		x = list(X).index(x0)
		stop = False
		i = 0
		
		init_belief = np.ones(nT) / nT
		
		while not stop:
			
			uct_node = MiuraLegibleMDP.MiuraMDPMCTSNode(nA, x, self, init_belief)
			a = uct_node.uct(self._goal_states, n_its, depth, beta, pol, verbose)
			x = rng_gen.choice(nX, p=P[A[a]][x, :])
			
			traj += [X[x]]
			actions += [A[a]]
			
			stop = (x in self._goal_states or i > 500)
			if stop:
				uct_node = MiuraLegibleMDP.MiuraMDPMCTSNode(nA, x, self, init_belief)
				actions += [A[uct_node.uct(self._goal_states, 1000, 20, 0.25, pol, verbose)]]
			
			i += 1
		
		return np.array(traj), np.array(actions)

	def trajectory_reward(self, trajs: np.ndarray, task_idx: int = None) -> float:
		
		r_ave = 0
		X = list(self.states)
		gamma = self.gamma
		n_trajs = len(trajs)
		t_idx = 1
		
		for traj in trajs:
			
			states = traj[0]
			actions = traj[1]
			traj_len = len(states)
			nT = len(self._tasks)
			curr_belief = np.ones(nT) / nT
			r_traj = 0
			g = 1
			
			for i in range(traj_len):
				
				x = X.index(states[i])
				a = actions[i]
				if i < traj_len - 2:
					x1 = X.index(states[i + 1])
				else:
					x1 = x
				
				curr_belief = self.update_belief(x, a, x1, curr_belief)
				r_traj += g * (- self.belief_reward(curr_belief[task_idx]))
				g *= gamma
			
			r_ave += r_traj / n_trajs
			if self._verbose:
				print('Miura trajectory: trajectory nr.%d! Finished %.2f%% of trajectories' % (t_idx, t_idx / n_trajs * 100), end='\n')
				t_idx += 1
	  
		if self._verbose:
			print('Miura trajectory reward finished!', end='\n')
	  
		return r_ave


class LearnerMDP(object):
	
	def __init__(self, x: np.ndarray, a: List, p: Dict, gamma: float, rewards: List, sign: int, verbose: bool):
		self._mdp_r = (x, a, p, gamma)
		self._pol = np.zeros((len(x), len(a)))
		self._reward = np.zeros((len(x), len(a)))
		self._reward_library = rewards
		self._sign = sign
		self._verbose = verbose
		pol_library = []
		q_library = []
		for i in range(len(rewards)):
			pol, q = self.policy_iteration(rewards[i])
			pol_library += [pol]
			q_library += [q]
		self._pol_library = np.array(pol_library)
		self._q_library = np.array(q_library)
	
	@property
	def mdp_r(self):
		return self._mdp_r
	
	@mdp_r.setter
	def mdp_r(self, mdp):
		self._mdp_r = mdp
	
	@property
	def pol_library(self):
		return self._pol_library
	
	@pol_library.setter
	def pol_library(self, pol):
		self._pol_library = pol
	
	@property
	def reward(self):
		return self._reward
	
	@reward.setter
	def reward(self, reward):
		self._reward = reward
	
	@property
	def reward_library(self):
		return self._reward_library
	
	@reward_library.setter
	def reward_library(self, rewards):
		self._reward_library = rewards
	
	def evaluate_pol(self, pol, c):
		X = self._mdp_r[0]
		A = self._mdp_r[1]
		P = self._mdp_r[2]
		gamma = self._mdp_r[3]
		if not c.any( ):
			c = self._reward
		
		nX = len(X)
		nA = len(A)
		
		# Cost and Probs averaged by policy
		cpi = (pol * c).sum(axis=1)
		ppi = pol[:, 0, None] * P[A[0]]
		for i in range(1, nA):
			ppi += pol[:, i, None] * P[A[i]]
		
		# J = (I - gamma*P)^-1 * c
		J = np.linalg.inv(np.eye(nX) - gamma * ppi).dot(cpi)
		
		return J
		
	def sample_probability(self, x: int, a: int, conf: float) -> np.ndarray:
		nR = len(self._reward_library)
		goals_likelihood = []
		
		for i in range(nR):
			q = self._q_library[i]
			goals_likelihood += [np.exp(self._sign * conf * (q[x, a] - np.max(q[x, :]))) / np.sum(np.exp(self._sign * conf * (q[x, :] - np.max(q[x, :]))))]
		
		goals_likelihood = np.array(goals_likelihood)
		return goals_likelihood
	
	def birl_inference(self, samples: np.ndarray, conf: float) -> Tuple[np.ndarray, int, float]:
		
		rng_gen = np.random.default_rng(int(time.time()))
		samples_likelihood = []
		n_tasks = len(self._reward_library)
		goal_prob = np.ones(n_tasks) / n_tasks
		
		for state, action in samples:
			sample_prob = self.sample_probability(state, action, conf)
			likelihood = goal_prob * sample_prob
			goal_prob += sample_prob
			goal_prob = goal_prob / goal_prob.sum()
			samples_likelihood += [likelihood]
		
		r_likelihood = np.cumprod(np.array(samples_likelihood), axis=0)[-1]
		p_max = r_likelihood / r_likelihood.sum( )
		reward_idx = rng_gen.choice(np.argwhere(p_max == np.amax(p_max)).ravel())
		reward_conf = p_max[reward_idx]
		
		return self._reward_library[reward_idx], reward_idx, reward_conf
	
	def birl_gradient_ascent(self, traj, conf, alpha):
		
		X = self._mdp_r[0]
		A = self._mdp_r[1]
		P = self._mdp_r[2]
		gamma = self._mdp_r[3]
		pol = self._pol_library
		c = self._reward
		nX = len(X)
		nA = len(A)
		nT = len(self._reward_library)
		goal_prob = np.ones(nT) / nT
		
		log_grad = np.zeros((nX, nA))
		ppi = pol[:, 0, None] * P[A[0]]
		for i in range(1, nA):
			ppi += pol[:, i, None] * P[A[i]]
		T_inv = np.linalg.inv(np.eye(nX) - gamma * ppi)
		
		for state, action in traj:
			sample_likelihood = goal_prob * self.sample_probability(state, action, conf)
			goal_prob = np.array(sample_likelihood)
			
			likelihood_q_derivative = conf * sample_likelihood * (1 - sample_likelihood)
			
			q_r_derivative = 1 + gamma * P[A[action]][state, :].dot(T_inv[:, state]) * pol[state, action]
			
			likelihood_grad = likelihood_q_derivative * q_r_derivative
			
			log_grad[state, action] += 1 / sample_likelihood * likelihood_grad
		
		self._reward = c + alpha * log_grad
	
	def policy_iteration(self, c):
		
		X = self._mdp_r[0]
		A = self._mdp_r[1]
		P = self._mdp_r[2]
		if not c.any( ):
			c = self._reward
		gamma = self._mdp_r[3]
		
		nS = len(X)
		nA = len(A)
		
		# Initialize pol
		pol = np.ones((nS, nA)) / nA
		
		# Initialize Q
		Q = np.zeros((nS, nA))
		
		quit = False
		i = 0
		
		while not quit:
			if self._verbose:
				print('Iteration %d' % (i + 1), end='\r')
			
			J = self.evaluate_pol(pol, c)
			
			for act in range(nA):
				Q[:, act, None] = c[:, act, None] + gamma * P[A[act]].dot(J[:, None])
			
			Qmax = Q.max(axis=1, keepdims=True)
			polnew = np.isclose(Q, Qmax, atol=1e-10, rtol=1e-10).astype(int)
			polnew = polnew / polnew.sum(axis=1, keepdims=True)
			
			quit = (pol == polnew).all( )
			pol = polnew
			i += 1
		
		if self._verbose:
			print('N. iterations: ', i)
		
		return pol, Q
	
	def trajectory(self, goal, pol, x0, rng_gen: np.random.Generator):
		
		X = self._mdp_r[0]
		A = self._mdp_r[1]
		P = self._mdp_r[2]
		
		nX = len(X)
		nA = len(A)
		
		traj = [x0]
		actions = []
		x = list(X).index(x0)
		stop = False
		
		while not stop:
			a = rng_gen.choice(nA, p=pol[x, :])
			x = rng_gen.choice(nX, p=P[A[a]][x, :])
			
			traj += [X[x]]
			actions += [A[a]]
			
			stop = (X[x].find(goal) != -1)
			if stop:
				actions += [A[rng_gen.choice(nA, p=pol[x, :])]]
		
		return np.array(traj), np.array(actions)
	
	def learner_eval(self, conf, trajs, traj_len, demo_step, goal):
		
		indexes = []
		n_trajs = len(trajs)
		it = 0
		for i in range(demo_step, traj_len + 1, demo_step):
			indexes += [i]
		
		if traj_len % demo_step == 0:
			n_idx = traj_len // demo_step
		else:
			n_idx = traj_len // demo_step + 1
			indexes += [traj_len]
		
		correct_count = np.zeros(n_idx)
		inference_conf = np.zeros(n_idx)
		for traj in trajs:
			for i in range(n_idx):
				idx = indexes[i]
				_, r_idx, r_conf = self.birl_inference(traj[:idx], conf)
				if r_idx == goal:
					correct_count[i] += 1
					inference_conf[i] += r_conf
			
			it += 1
			if self._verbose:
				print('Completed %d%% of trajectories' % (int(it / n_trajs * 100)), end='\r')
		
		avg_inference_conf = []
		for i in range(n_idx):
			if correct_count[i] != 0:
				avg_inference_conf += [inference_conf[i] / correct_count[i]]
			else:
				avg_inference_conf += [0.0]
		
		if self._verbose:
			print('Finished.')
		return correct_count, np.array(avg_inference_conf)


class OPoLMDP(MDP):
	
	def __init__(self, X: np.ndarray, A: np.ndarray, P: Dict[str, np.ndarray], gamma: float, Z: List[str], O: Dict[str, np.ndarray], verbose: bool, task: str,
				 task_states: List[Tuple[int, int, str]], tasks: List[str], beta: float, goal_states: List[int], sign: int, q_mdps: Dict[str, np.ndarray]):
		self._task = task
		self._tasks = tasks
		self._Z = Z
		self._O = O
		self._task_states = {}
		for task_state in task_states:
			self._task_states[task_state[2]] = tuple([task_state[0], task_state[1]])
		self._tasks_q = q_mdps
		self._beta = beta
		nX = len(X)
		nA = len(A)
		nT = len(tasks)
		c = np.zeros((nT, nX, nA))
		
		super().__init__(X, A, P, c, gamma, goal_states, 'rewards', verbose)
		for t in range(nT):
			for x in range(nX):
				for a in range(nA):
					c[t, x, a] = self.obstructed_legible_reward(x, a, t, sign, nX, nA, P, A)
		
		self._mdp = (X, A, P, c, gamma)
	
	def update_cost_function(self, sign: int, q_mdps: Dict[str, np.ndarray]):
		mdp = self.mdp
		X = mdp[0]
		A = mdp[1]
		P = mdp[2]
		gamma = mdp[4]
		nX = len(X)
		nA = len(A)
		nT = len(self._tasks)
		c = np.zeros((nT, nX, nA))
		self._tasks_q = q_mdps
		
		for t in range(nT):
			for x in range(nX):
				for a in range(nA):
					c[t, x, a] = self.obstructed_legible_reward(x, a, t, sign, nX, nA, P, A)
		
		self._mdp = (X, A, P, c, gamma)
		
	def obstructed_legible_reward(self, x_idx: int, a_idx: int, task: int, sign: int, nX: int, nA: int, P: Dict[str, np.ndarray], A: List[str]) -> float:
		
		leg_reward = 0.0
		for z in range(len(self._Z)):
			leg_reward += self.observation_goal(z, task, sign, nX, nA, P, A) * self.observation_state_action(x_idx, z, nX, P, A[a_idx])
		
		return leg_reward
	
	def observation_state_action(self, x_idx: int, z: int, nX: int, P: Dict[str, np.ndarray], a: str) -> float:
	
		obs_prob = 0.0
		for x1 in range(nX):
			obs_prob += self._O[a][x1, z] * P[a][x_idx, x1]
		
		return obs_prob
	
	def observation_goal(self, z: int, t: int, sign: int, nX: int, nA: int, P: Dict[str, np.ndarray], A: List[str]) -> float:
	
		obs_prob = 0.0
		for x in range(nX):
			for a in range(nA):
				obs_prob += self.observation_state_action(x, z, nX, P, A[a]) * self.task_probability(x, a, t, sign)
			
		return obs_prob
	
	def task_probability(self, x: int, a: int, task: int, sign: int) -> float:
		
		task_cost = np.exp(sign * self._beta * self._tasks_q[self._tasks[task]][x, a])
		tasks_sum = task_cost
		for t in self._tasks:
			if t != self._tasks[task]:
				tasks_sum += np.exp(sign * self._beta * self._tasks_q[t][x, a])
		
		return task_cost / tasks_sum
		

class TeamMDP(MDP):
	
	def __init__(self, team_info: Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray], robot_pols: Dict[str, np.ndarray], gamma: float,
				 feedback_type: str, verbose: bool, team_world: MDPWorld, n_robots: int, task: str, obj_states: List[Tuple[int, int, str]],
				 goal_states: List[int], fail_chance: float, robot_states: np.ndarray, robot_actions: np.ndarray, robot_transitions: Dict[str, np.ndarray]):
		
		self._task = task
		self._seq_len = len(task)
		self._fail_chance = fail_chance
		self._obj_states = {}
		self._robot_pols = {}
		for task_state in obj_states:
			self._obj_states[task_state[2]] = tuple([task_state[0], task_state[1]])
			self._robot_pols[task_state[2]] = robot_pols[task_state[2]]
		self._team_world = team_world
		self._robot_states = robot_states
		self._robot_actions = robot_actions
		self._robot_transitions = robot_transitions
		self._n_robots = n_robots
		
		x_t, a_t, P_t, c_t = team_info
		super().__init__(x_t, a_t, P_t, c_t, gamma, goal_states, feedback_type, verbose)
	
	def team_decision(self, team_x0: str, team_pol: np.ndarray, rng_gen: np.random.Generator, robots_x0: List[str],
					  max_iterations: int) -> (Tuple[List[str], List[str]], Tuple[List[List[str]], List[List[str]]]):
		
		team_states = []
		team_actions = []
		robots_states = []
		robots_actions = []
		robots_controls = ['F'] * self._n_robots
		X_team, A_team, P_team, _, _ = self._mdp
		world_team_state = Utilities.state_str_to_tuple(team_x0)
		seq_state = world_team_state[-1][0]
		goal_seq = Utilities.state_str_to_tuple(X_team[self._goal_states[0]])[-1][0]
		robot_states_lst = list(self._robot_states)
		team_states_lst = list(X_team)
		
		nA_team = len(A_team)
		
		for i in range(self._n_robots):
			robots_states += [[robots_x0[i]]]
			robots_actions += [[]]
	
		team_states += [team_x0]
		team_x = team_states_lst.index(team_x0)
		finish = False
		i = 0
		
		while not finish:
			
			team_action = rng_gen.choice(nA_team, p=team_pol[team_x, :])
			team_actions += [A_team[team_action]]
			action_detailed = self._team_world.action_detailed(A_team[team_action])
			action_results = rng_gen.choice([0, 1], p=[self._fail_chance, 1 - self._fail_chance])
			
			if action_results > 0:
				if action_detailed[0] == 'U':
					robots_controls[action_detailed[2] - 1] = ''.join(action_detailed[1])
				
				seq_state_tmp = seq_state
				for i in range(self._n_robots):
					if robots_controls[i].find('F') == -1:
						robot_target = robots_controls[i]
						if seq_state_tmp.find(robot_target) == -1:
							if goal_seq[:len(seq_state_tmp)+1] == (seq_state_tmp + robot_target):
								x_r = robots_states[i][-1] + ', 1'
							else:
								x_r = robots_states[i][-1] + ', 0'
							a_r = rng_gen.choice(self._robot_actions, p=self._robot_pols[robot_target][robot_states_lst.index(x_r), :])
							x_r_1 = rng_gen.choice(self._robot_states, p=self._robot_transitions[a_r][robot_states_lst.index(x_r), :])
							if x_r_1.find(robot_target) != -1:
								if len(seq_state_tmp) < self._seq_len:
									seq_state_tmp += robot_target
								robots_controls[i] = 'F'
							
							robots_actions[i] += [a_r]
							robots_states[i] += [Utilities.state_tuple_to_str(Utilities.state_str_to_tuple(x_r_1)[:-1])]
						
						else:
							robots_controls[i] = 'F'
							robots_actions[i] += ['N']
							robots_states[i] += [robots_states[i][-1]]
						
					else:
						robots_actions[i] += ['N']
						robots_states[i] += [robots_states[i][-1]]
				
				seq_state = seq_state_tmp
				team_x = team_states_lst.index(Utilities.state_tuple_to_str(tuple([(str(x[0]), ) for x in robots_controls]) + ((seq_state, ), )))
			
			team_states += [X_team[team_x]]
			finish = (team_x in self._goal_states or i >= max_iterations)
			if finish:
				team_actions += [A_team[rng_gen.choice(nA_team, p=team_pol[team_x, :])]]
		
		return (team_states, team_actions), (robots_states, robots_actions)
	
	
class TeamLegibleMDP(MDP):
	
	def __init__(self, team_info: Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]], gamma: float, verbose: bool, task: str, tasks: List[str],
				 obj_states: List[Tuple[int, int, str]], beta: float, goal_states: List[int], sign: int, q_mdps: Dict[str, np.ndarray], team_world: MDPWorld,
				 n_robots: int, robot_pols: Dict[str, np.ndarray], fail_chance: float, robot_states: np.ndarray, robot_actions: np.ndarray,
				 robot_transitions: Dict[str, np.ndarray]):
		
		self._task = task
		self._tasks = tasks
		self._beta = beta
		self._seq_len = len(task)
		self._fail_chance = fail_chance
		self._obj_states = {}
		self._robot_pols = {}
		for task_state in obj_states:
			self._obj_states[task_state[2]] = tuple([task_state[0], task_state[1]])
			self._robot_pols[task_state[2]] = robot_pols[task_state[2]]
		self._tasks_q = q_mdps
		self._team_world = team_world
		self._robot_states = robot_states
		self._robot_actions = robot_actions
		self._robot_transitions = robot_transitions
		self._n_robots = n_robots
		
		x_t, a_t, p_t = team_info
		nX = len(x_t)
		nA = len(a_t)
		nT = len(tasks)
		c = np.zeros((nT, nX, nA))
		
		super().__init__(x_t, a_t, p_t, c, gamma, goal_states, 'rewards', verbose)
		for t in range(nT):
			for i in range(nX):
				for j in range(nA):
					c[t, i, j] = self.legible_reward(i, j, t, sign)
		
		self._mdp = (x_t, a_t, p_t, c, gamma)
	
	def update_cost_function(self, sign: int, q_mdps: Dict[str, np.ndarray]):
		mdp = self.mdp
		X = mdp[0]
		A = mdp[1]
		P = mdp[2]
		gamma = mdp[4]
		nX = len(X)
		nA = len(A)
		nT = len(self._tasks)
		c = np.zeros((nT, nX, nA))
		self._tasks_q = q_mdps
		
		for t in range(nT):
			for x in range(nX):
				for a in range(nA):
					c[t, x, a] = self.legible_reward(x, a, t, sign, nX, nA, P, A)
		
		self._mdp = (X, A, P, c, gamma)
		
	def legible_reward(self, x_idx: int, a_idx: int, task: int, sign: int) -> float:
		
		task_cost = np.exp(sign * self._beta * self._tasks_q[self._tasks[task]][x_idx, a_idx])
		tasks_sum = task_cost
		for t in self._tasks:
			if t != self._tasks[task]:
				tasks_sum += np.exp(sign * self._beta * self._tasks_q[t][x_idx, a_idx])
		
		return task_cost / tasks_sum
	
	def team_decision(self, team_x0: str, team_pol: np.ndarray, rng_gen: np.random.Generator, robots_x0: List[str],
					  max_iterations: int) -> (Tuple[List[str], List[str]], Tuple[List[List[str]], List[List[str]]]):
		team_states = []
		team_actions = []
		robots_states = []
		robots_actions = []
		robots_controls = ['F'] * self._n_robots
		X_team, A_team, P_team, _, _ = self._mdp
		world_team_state = Utilities.state_str_to_tuple(team_x0)
		seq_state = world_team_state[-1][0]
		goal_seq = Utilities.state_str_to_tuple(X_team[self._goal_states[0]])[-1][0]
		robot_states_lst = list(self._robot_states)
		team_states_lst = list(X_team)
		
		nA_team = len(A_team)
		
		for i in range(self._n_robots):
			robots_states += [[robots_x0[i]]]
			robots_actions += [[]]
		
		team_states += [team_x0]
		team_x = team_states_lst.index(team_x0)
		finish = False
		i = 0
		
		while not finish:
			team_action = rng_gen.choice(nA_team, p=team_pol[team_x, :])
			team_actions += [A_team[team_action]]
			action_detailed = self._team_world.action_detailed(A_team[team_action])
			action_results = rng_gen.choice([0, 1], p=[self._fail_chance, 1 - self._fail_chance])
			
			if action_results > 0:
				if action_detailed[0] == 'U':
					robots_controls[action_detailed[2] - 1] = ''.join(action_detailed[1])
				
				seq_state_tmp = seq_state
				for i in range(self._n_robots):
					if robots_controls[i].find('F') == -1:
						robot_target = robots_controls[i]
						if seq_state_tmp.find(robot_target) == -1:
							if goal_seq[:len(seq_state_tmp) + 1] == (seq_state_tmp + robot_target):
								x_r = robots_states[i][-1] + ', 1'
							else:
								x_r = robots_states[i][-1] + ', 0'
							a_r = rng_gen.choice(self._robot_actions, p=self._robot_pols[robot_target][robot_states_lst.index(x_r), :])
							x_r_1 = rng_gen.choice(self._robot_states, p=self._robot_transitions[a_r][robot_states_lst.index(x_r), :])
							if x_r_1.find(robot_target) != -1:
								if len(seq_state_tmp) < self._seq_len:
									seq_state_tmp += robot_target
								robots_controls[i] = 'F'
							
							robots_actions[i] += [a_r]
							robots_states[i] += [Utilities.state_tuple_to_str(Utilities.state_str_to_tuple(x_r_1)[:-1])]
					
						else:
							robots_controls[i] = 'F'
							robots_actions[i] += ['N']
							robots_states[i] += [robots_states[i][-1]]
					
					else:
						robots_actions[i] += ['N']
						robots_states[i] += [robots_states[i][-1]]
				
				seq_state = seq_state_tmp
				team_x = team_states_lst.index(Utilities.state_tuple_to_str(tuple([(str(x[0]),) for x in robots_controls]) + ((seq_state,),)))
			
			team_states += [X_team[team_x]]
			finish = (team_x in self._goal_states or i >= max_iterations)
			if finish:
				team_actions += [A_team[rng_gen.choice(nA_team, p=team_pol[team_x, :])]]
		
		return (team_states, team_actions), (robots_states, robots_actions)
	
