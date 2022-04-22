#! /usr/bin/env python

from __future__ import annotations
import numpy as np
import time
import math
from termcolor import colored
from typing import List, Dict, Tuple


class Utilities(object):
	
	@staticmethod
	def q_from_pol(mdp: HighDimMDP, pol: np.ndarray, task_idx: int = None) -> np.ndarray:
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
			p = np.zeros((nX, nX))
			for state_idx in range(nX):
				state_tansitions = P[A[act]][state_idx]
				for state in state_tansitions:
					p[state_idx, int(state[0])] += state[1]
			q[:, act, None] = c[:, act, None] + gamma * p.dot(J)
		
		return q
	
	@staticmethod
	def v_from_q(q: np.ndarray, pol: np.ndarray) -> np.ndarray:
		return (q * pol).sum(axis=1)
	
	@staticmethod
	def Ppi(transitions: Dict[str, np.ndarray], actions: List[str], pol: np.ndarray) -> np.ndarray:
		nX = len(transitions[actions[0]])
		nA = len(actions)
		ppi = np.zeros((nX, nX))
		for act_idx in range(nA):
			for state_idx in range(nX):
				state_tansitions = transitions[actions[act_idx]][state_idx]
				for state in state_tansitions:
					ppi[state_idx, int(state[0])] += pol[state_idx, act_idx] * state[1]
		
		return ppi
	
	@staticmethod
	def softmax(param: np.ndarray, temp: float) -> np.ndarray:
		tmp = np.exp((param - np.max(param, axis=1)[:, None]) / temp)
		return tmp / tmp.sum(axis=1)[:, None]
	
	@staticmethod
	def softmax_grad(param: np.ndarray, temp: float) -> np.ndarray:
		softmax_pol = Utilities.softmax(param, temp)
		nX, nA = softmax_pol.shape
		sofmax_grad = np.zeros((nX * nA, nX * nA))
		
		for x in range(nX):
			for a in range(nA):
				for a2 in range(nA):
					sofmax_grad[x * a, x * a2] = softmax_pol[x, a] * ((1 if a == a2 else 0) - softmax_pol[x, a2])
		
		return np.diagonal(sofmax_grad).reshape((nX, nA), order='F')

	@staticmethod
	def get_goal_state(states: np.ndarray, goal: Tuple, obj_locs: List[Tuple]) -> List[int]:
		state_lst = list(states)
		goal_loc = ''
		for obj_loc in obj_locs:
			if obj_loc[2] in goal:
				goal_loc += str(obj_loc[0]) + ' ' + str(obj_loc[1]) + ', '
		return [state_lst.index(x) for x in states if x.find(goal_loc[:-2]) != -1]


class HighDimMDP(object):
	
	def __init__(self, x: np.ndarray, a: List[str], p: Dict[str, np.ndarray], c: np.ndarray, gamma: float, goal_states: List[int], feedback_type: str,
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
	def actions(self) -> List[str]:
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
	def mdp(self) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray], np.ndarray, float]:
		return self._mdp
	
	@property
	def verbose(self) -> bool:
		return self._verbose
	
	@mdp.setter
	def mdp(self, mdp: Tuple[np.ndarray, List[str], Dict[str, np.ndarray], np.ndarray, float]):
		self._mdp = mdp
	
	@goals.setter
	def goals(self, goals: List[str]):
		self._goal_states = goals
	
	@verbose.setter
	def verbose(self, verbose: bool):
		self._verbose = verbose
	
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
		
		nX = len(X)
		nA = len(A)
		
		J = np.zeros(nX)
		err = 1
		i = 0
		
		while err > 1e-8:
			Q = []
			for act in range(nA):
				p = np.zeros((nX, nX))
				for state_idx in range(nX):
					state_tansitions = P[A[act]][state_idx]
					for state in state_tansitions:
						p[state_idx, int(state[0])] += state[1]
				Q += [c[:, act] + gamma * p.dot(J)]
			
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
		
		nX = len(X)
		nA = len(A)
		
		# Initialize pol
		pol = np.ones((nX, nA)) / nA
		
		# Initialize Q
		Q = np.zeros((nX, nA))
		
		quit = False
		i = 0
		
		while not quit:
			if self._verbose:
				print('Iteration %d' % (i + 1), end='\r')
			
			J = self.evaluate_pol(pol, task_idx)
			
			for act in range(nA):
				p = np.zeros((nX, nX))
				for state_idx in range(nX):
					state_tansitions = P[A[act]][state_idx]
					for transition in state_tansitions:
						p[state_idx, int(transition[0])] += transition[1]
				Q[:, act, None] = c[:, act, None] + gamma * p.dot(J)
			
			if self._feedback_type.find('cost') != -1:
				Qbest = Q.min(axis=1, keepdims=True)
			else:
				Qbest = Q.max(axis=1, keepdims=True)
			polnew = np.isclose(Q, Qbest, atol=1e-10, rtol=1e-10).astype(int)
			polnew = polnew / polnew.sum(axis=1, keepdims=True)
			
			quit = (pol == polnew).all()
			pol = polnew
			i += 1
		
		if self._verbose:
			print('N. iterations: ', i)
		
		return pol, Q
	
	def trajectory_len(self, x0: str, pol: np.ndarray, traj_len: int, rng_gen: np.random.Generator) -> (np.ndarray, np.ndarray):
		X = self._mdp[0]
		A = self._mdp[1]
		P = self._mdp[2]
		
		nA = len(A)
		
		traj = [x0]
		actions = []
		x = list(X).index(x0)
		
		for _ in range(traj_len):
			a = rng_gen.choice(nA, p=pol[x, :])
			next_states = P[A[a]][x, :]
			x = rng_gen.choice(next_states[:, :, 0], p=next_states[:, :, 1])
			
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
			nxt_states = P[A[a]][x, :]
			x = int(rng_gen.choice(nxt_states[:, 0], p=nxt_states[:, 1]))
			
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
								next_states = P[A[pol_act[j]]][x, :]
								x_tmp = rng_gen.choice(next_states[:, :, 0], p=next_states[:, :, 1])
								while x_tmp == x:
									x_tmp = rng_gen.choice(nX, p=P[A[pol_act[j]]][x, :])
							else:
								next_states = P[A[pol_act[j]]][x, :]
								x_tmp = rng_gen.choice(next_states[:, :, 0], p=next_states[:, :, 1])
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


class HighDimLegibleMDP(HighDimMDP):
	
	def __init__(self, x: np.ndarray, a: List[str], p: Dict[str, np.ndarray], gamma: float, verbose: bool, task: str, task_states: List[Tuple[int, int, str]],
				 tasks: List[str], beta: float, goal_states: List[int], sign: int, q_mdps: Dict[str, np.ndarray]):
		self._task = task
		self._tasks = tasks
		self._task_states = {}
		for task_state in task_states:
			self._task_states[task_state[2]] = tuple([task_state[0], task_state[1]])
		self._tasks_q = q_mdps
		self._beta = beta
		nX = len(x)
		nA = len(a)
		nT = len(tasks)
		c = np.zeros((nT, nX, nA))
		
		super().__init__(x, a, p, c, gamma, goal_states, 'rewards', verbose)
		for t in range(nT):
			for i in range(nX):
				for j in range(nA):
					c[t, i, j] = self.optimal_legible_cost(i, j, t, sign)
		
		self._mdp = (x, a, p, c, gamma)
	
	def update_cost_function(self, sign: int):
		mdp = self.mdp
		x = mdp[0]
		a = mdp[1]
		p = mdp[2]
		gamma = mdp[4]
		nX = len(x)
		nA = len(a)
		nT = len(self._tasks)
		c = np.zeros((nT, nX, nA))
		
		for t in range(nT):
			for i in range(nX):
				for j in range(nA):
					c[t, i, j] = self.optimal_legible_cost(i, j, t, sign)
		
		self._mdp = (x, a, p, c, gamma)
	
	def optimal_legible_cost(self, x: int, a: int, t: int, sign: int) -> float:
		task_cost = np.exp(sign * self._beta * self._tasks_q[self._tasks[t]][x, a])
		tasks_sum = task_cost
		for task in self._tasks:
			if task != self._tasks[t]:
				tasks_sum += np.exp(sign * self._beta * self._tasks_q[task][x, a])
		
		return task_cost / tasks_sum
	
	def legible_cost(self, x: int, a: int, t: int, sign: int, dist: np.ndarray) -> float:
		task_cost = np.exp(sign * self._beta * self._tasks_q[self._tasks[t]][x, a]) / (1 / dist[t, x])
		tasks_sum = task_cost
		for task in self._tasks:
			if task != self._tasks[t]:
				tasks_sum += np.exp(sign * self._beta * self._tasks_q[task][x, a]) / (1 / dist[self._tasks.index(task), x])
		
		return task_cost / tasks_sum
	
