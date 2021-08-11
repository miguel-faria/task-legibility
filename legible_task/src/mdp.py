#! /usr/bin/env python

import numpy as np
import time
import math
import re
from tqdm import tqdm
from termcolor import colored
from typing import List, Dict, Tuple


class Utilities(object):

	@staticmethod
	def v_from_q(q: np.ndarray, pol: np.ndarray) -> np.ndarray:
		return (q * pol).sum(axis=1)
	
	# @staticmethod
	# def q_from_v(v: np.ndarray, pol: np.ndarray) -> np.ndarray:
	
	@staticmethod
	def Ppi(transitions: Dict[str, np.ndarray], actions: List[str], pol: np.ndarray) -> np.ndarray:

		nA = len(actions)
		ppi = pol[:, 0, None] * transitions[actions[0]]
		for i in range(1, nA):
			ppi += pol[:, i, None] * transitions[actions[i]]
			
		return ppi

	@staticmethod
	def Ppi_stack(transitions: Dict[str, np.ndarray], actions: List[str], pol: np.ndarray) -> np.ndarray:
		
		nX = len(transitions[actions[0]])
		nA = len(actions)
		ppi = np.zeros((nX*nA, nX*nA))
		
		for a in range(nA):
			for a2 in range(nA):
				ppi[a*nX:(a+1)*nX, a2*nX:(a2+1)*nX] = transitions[actions[a]] * pol[:, a2, None]
		
		return ppi

	@staticmethod
	def Q_func_grad(transitions: Dict[str, np.ndarray], actions: List[str], pol: np.ndarray, q_pi: np.ndarray, gamma: float) -> np.ndarray:
		
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
	
	def __init__(self, x: np.ndarray, a: List[str], p: Dict[str, np.ndarray], c: np.ndarray, gamma: float, goal_states: List[str], feedback_type: str):
		self._mdp = (x, a, p, c, gamma)
		self._goal_states = goal_states
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
	def goals(self) -> List[str]:
		return self._goal_states
	
	@property
	def mdp(self) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray], np.ndarray, float]:
		return self._mdp
	
	@mdp.setter
	def mdp(self, mdp: Tuple[np.ndarray, List[str], Dict[str, np.ndarray], np.ndarray, float]):
		self._mdp = mdp
	
	@goals.setter
	def goals(self, goals: List[str]):
		self._goal_states = goals
	
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
		
		print('N. iterations: ', i)
		
		return pol, Q
	
	def trajectory_len(self, x0: str, pol: np.ndarray, traj_len: int) -> (np.ndarray, np.ndarray):
		X = self._mdp[0]
		A = self._mdp[1]
		P = self._mdp[2]
		
		nX = len(X)
		nA = len(A)
		
		traj = [x0]
		actions = []
		x = list(X).index(x0)
		
		for _ in range(traj_len):
			a = np.random.choice(nA, p=pol[x, :])
			x = np.random.choice(nX, p=P[A[a]][x, :])
			
			traj += [X[x]]
			actions += [A[a]]
		
		actions += [A[np.random.choice(nA, p=pol[x, :])]]
		return np.array(traj), np.array(actions)
	
	def trajectory(self, x0: str, pol: np.ndarray) -> (np.ndarray, np.ndarray):
		X = self._mdp[0]
		A = self._mdp[1]
		P = self._mdp[2]
		
		nX = len(X)
		nA = len(A)
		
		traj = [x0]
		actions = []
		x = list(X).index(x0)
		stop = False
		
		while not stop:
			a = np.random.choice(nA, p=pol[x, :])
			x = np.random.choice(nX, p=P[A[a]][x, :])
			
			traj += [X[x]]
			actions += [A[a]]
			
			stop = (X[x] in self._goal_states)
			if stop:
				actions += [A[np.random.choice(nA, p=pol[x, :])]]
		
		return np.array(traj), np.array(actions)
	
	def all_trajectories(self, x0:str, pol: np.ndarray) -> (np.ndarray, np.ndarray):
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
					a_traj += [A[np.random.choice(nA, p=pol[x, :])]]
					add_traj = True
					break
				
				else:
					pol_act = np.nonzero(pol[x, :])[0]
					if len(pol_act) > 1:
						for j in range(1, len(pol_act)):
							if len(np.nonzero(P[A[pol_act[j]]][x, :])) > 1:
								x_tmp = np.random.choice(nX, p=P[A[pol_act[j]]][x, :])
								while x_tmp == x:
									x_tmp = np.random.choice(nX, p=P[A[pol_act[j]]][x, :])
							else:
								x_tmp = np.random.choice(nX, p=P[A[pol_act[j]]][x, :])
							tmp_traj = [list(traj) + [X[x_tmp]], list(a_traj) + [A[pol_act[j]]]]
							if tmp_traj not in started_trajs:
								started_trajs += [tmp_traj]
					
					a = pol_act[0]
					x = np.random.choice(nX, p=P[A[a]][x, :])
					
					if X[x] != traj[-1]:
						traj += [X[x]]
						a_traj += [A[a]]
						
						stop_inner = (x in self._goal_states)
						if stop_inner:
							a_traj += [A[np.random.choice(nA, p=pol[x, :])]]
						
						add_traj = True
					
					else:
						if it > 1000:
							stop_inner = True
					
					it += 1
			
			i += 1
			stop = (i >= len(started_trajs) or i > 1000)
			if add_traj:
				trajs += [np.array(traj)]
				acts += [np.array(a_traj)]
		
		return np.array(trajs, dtype=object), np.array(acts, dtype=object)
	
	def trajectory_reward(self, trajs: np.ndarray) -> float:
		r_avg = 0
		X = list(self._mdp[0])
		A = list(self._mdp[1])
		c = self._mdp[3]
		gamma = self._mdp[4]
		n_trajs = len(trajs)
		
		with tqdm(total=len(trajs)) as progress_bar:
			for traj in trajs:
				states = traj[0]
				actions = traj[1]
				r_traj = 0
				g = 1
				
				for j in range(len(actions)):
					x = states[j]
					a = actions[j]
					r_traj += g * c[X.index(x), A.index(a)]
					g *= gamma
				
				r_avg += r_traj / n_trajs
				progress_bar.update(1)
		
		return r_avg


class LegibleTaskMDP(MDP):
	
	def __init__(self, x: np.ndarray, a: List[str], p: Dict[str, np.ndarray], gamma: float, task: str, task_states: List[Tuple[int, int, str]], tasks: List[str],
				 beta: float, goal_states: List[str], sign: int, leg_func: str, q_mdps: Dict[str, np.ndarray], v_mdps: Dict[str, np.ndarray]):
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
		
		super().__init__(x, a, p, c, gamma, goal_states, 'rewards')
		if leg_func in list(self._legible_functions.keys()):
			for t in range(nT):
				for i in range(nX):
					for j in range(nA):
						c[t, i, j] = self._legible_functions[leg_func](i, j, t, sign)
			
			self._mdp = (x, a, p, c, gamma)
		
		else:
			print(colored('Invalid legibility function. Exiting without computing cost function.',
						  'red'))
			return
	
	def update_cost_function(self, legible_func: str, sign: int):
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
						c[t, i, j] = self._legible_functions[legible_func](i, j, t, sign)
			
			self._mdp = (x, a, p, c, gamma)
		
		else:
			print(colored('Invalid legibility function. Exiting without computing cost function.',
						  'red'))
			return
	
	def optimal_legible_cost(self, x: int, a: int, t: int, sign: int) -> float:
		task_cost = np.exp(sign * self._beta * self._tasks_q[self._tasks[t]][x, a])
		tasks_sum = task_cost
		for task in self._tasks:
			if task != self._tasks[t]:
				tasks_sum += np.exp(sign * self._beta * self._tasks_q[task][x, a])
		
		return task_cost / tasks_sum
	
	def legible_cost(self,  x: int, a: int, t: int, sign: int) -> float:
		X = self._mdp[0]
		A = self._mdp[1]
		P = self._mdp[2]
		
		sa_prob = self.optimal_legible_cost(x, a, t, sign)
		
		nxt_states = np.nonzero(P[A[a]][x, :])[0]
		
		x_dist_ratio = 0
		
		for state in nxt_states:
			state_prob = P[A[a]][x, state]
			
			state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", X[state], re.I)
			row = int(state_split.group(1))
			col = int(state_split.group(2))
			
			task_dist = math.inf
			for task in self._task_states.keys( ):
				task_state = self._task_states[task]
				dist = math.sqrt((row - task_state[0]) ** 2 + (col - task_state[1]) ** 2)
				if dist < task_dist:
					task_dist = dist
			
			goal_dist = math.inf
			for goal in self._tasks[t]:
				dist = math.sqrt((row - self._task_states[goal][0]) ** 2 + (col - self._task_states[goal][1]) ** 2)
				if dist < goal_dist:
					goal_dist = dist
			
			try:
				x_dist_ratio += state_prob * min(task_dist / goal_dist, 1.0)
			except ZeroDivisionError:
				x_dist_ratio += state_prob
		
		return sa_prob * x_dist_ratio

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
			
			# print('Iteration: %d\t Legibility: %.8f' % (it+1, legibility), end='\r')
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
		
		print('Took %d iterations\t Legibility: %.8f' % (it + 1, legibility))
		
		if best_action:
			pol_best = pol.max(axis=1, keepdims=True)
			pol = np.isclose(pol, pol_best, atol=1e-10, rtol=1e-10).astype(int)
			pol = pol / pol.sum(axis=1, keepdims=True)
		
		return pol


class LearnerMDP(object):
	
	def __init__(self, x, a, p, gamma, rewards, sign):
		self._mdp_r = (x, a, p, gamma)
		self._pol = np.zeros((len(x), len(a)))
		self._reward = np.zeros((len(x), len(a)))
		self._reward_library = rewards
		self._sign = sign
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
	
	def likelihood(self, x, a, conf):
		A = self._mdp_r[1]
		P = self._mdp_r[2]
		gamma = self._mdp_r[3]
		rewards = self._reward_library
		pols = self._pol_library
		nR = len(rewards)
		likelihood = []
		
		for i in range(nR):
			c = rewards[i]
			J = self.evaluate_pol(pols[i], c)
			q_star = np.zeros(len(A))
			for act_idx in range(len(A)):
				q_star[act_idx] = c[x, act_idx] + gamma * P[A[act_idx]][x, :].dot(J)
			
			likelihood += [np.exp(self._sign * conf * q_star[a]) / np.sum(np.exp(self._sign * conf * q_star))]
		
		return likelihood
	
	def birl_inference(self, traj, conf):
		likelihoods = []
		
		for state, action in traj:
			likelihood = []
			for i in range(len(self._reward_library)):
				q = self._q_library[i]
				likelihood += [np.exp(self._sign * conf * q[state, action]) /
							   np.sum(np.exp(self._sign * conf * q[state, :]))]
			likelihoods += [likelihood]
		
		r_likelihood = np.cumprod(np.array(likelihoods), axis=0)[-1]
		max_likelihood = np.max(r_likelihood)
		low_magnitude = math.floor(math.log(np.min(r_likelihood), 10)) - 2
		p_max = np.isclose(r_likelihood, max_likelihood, atol=10 ** low_magnitude, rtol=10 ** low_magnitude).astype(int)
		p_max = p_max / p_max.sum( )
		reward_idx = np.random.choice(len(self._reward_library), p=p_max)
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
		
		log_grad = np.zeros((nX, nA))
		ppi = pol[:, 0, None] * P[A[0]]
		for i in range(1, nA):
			ppi += pol[:, i, None] * P[A[i]]
		T_inv = np.linalg.inv(np.eye(nX) - gamma * ppi)
		
		for state, action in traj:
			sa_likelihood = self.likelihood(state, action, conf)
			
			likelihood_q_derivative = conf * sa_likelihood * (1 - sa_likelihood)
			
			q_r_derivative = 1 + gamma * P[A[action]][state, :].dot(T_inv[:, state]) * pol[state, action]
			
			likelihood_grad = likelihood_q_derivative * q_r_derivative
			
			log_grad[state, action] += 1 / sa_likelihood * likelihood_grad
		
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
		
		print('N. iterations: ', i)
		
		return pol, Q
	
	def trajectory(self, goal, pol, x0):
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
			a = np.random.choice(nA, p=pol[x, :])
			x = np.random.choice(nX, p=P[A[a]][x, :])
			
			traj += [X[x]]
			actions += [A[a]]
			
			stop = (X[x].find(goal) != -1)
			if stop:
				actions += [A[np.random.choice(nA, p=pol[x, :])]]
		
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
				reward, r_idx, r_conf = self.birl_inference(traj[:idx], conf)
				if r_idx == goal:
					correct_count[i] += 1
					inference_conf[i] += r_conf
			
			it += 1
			print('Completed %d%% of trajectories' % (int(it / n_trajs * 100)), end='\r')
		
		avg_inference_conf = []
		for i in range(n_idx):
			if correct_count[i] != 0:
				avg_inference_conf += [inference_conf[i] / correct_count[i]]
			else:
				avg_inference_conf += [0.0]
		
		print('Finished.')
		return correct_count, avg_inference_conf


def main( ):
	from mazeworld import AutoCollectMazeWord, LimitedCollectMazeWorld
	
	def get_goal_states(states, goal):
		state_lst = list(states)
		return [state_lst.index(x) for x in states if x.find(goal) != -1]
	
	def simulate(mdp, pol, mdp_tasks, leg_pol, x0, n_trajs):
		mdp_trajs = []
		tasks_trajs = []
		
		for _ in tqdm(range(n_trajs), desc='Simulate Trajectories'):
			traj, acts = mdp.trajectory(x0, pol)
			traj_leg, acts_leg = mdp_tasks.trajectory(x0, leg_pol)
			mdp_trajs += [[traj, acts]]
			tasks_trajs += [[traj_leg, acts_leg]]
		
		mdp_r = mdp.trajectory_reward(mdp_trajs)
		mdp_rl = mdp_tasks.trajectory_reward(mdp_trajs)
		task_r = mdp.trajectory_reward(tasks_trajs)
		task_rl = mdp_tasks.trajectory_reward(tasks_trajs)
		
		return mdp_r, mdp_rl, task_r, task_rl
	
	n_rows = 8
	n_cols = 10
	# objs_states = [(7, 2, 'P'), (4, 9, 'D'), (2, 7, 'C')]
	objs_states = [(7, 2, 'P'), (4, 9, 'D'), (2, 7, 'C'), (2, 4, 'L'), (5, 5, 'T'), (8, 6, 'O')]
	# x0 = np.random.choice([x for x in X_a if 'N' in x])
	x0 = '1 1 N'
	# goals = ['P', 'D', 'C']
	goals = ['P', 'D', 'C', 'L', 'T', 'O']
	goal = 'D'
	
	print('Initial State: ' + x0)
	print('######################################')
	print('#####   Auto Collect Maze World  #####')
	print('######################################')
	print('### Generating World ###')
	acmw = AutoCollectMazeWord( )
	X_a, A_a, P_a = acmw.generate_world(n_rows, n_cols, objs_states)
	
	print('### Computing Costs and Creating Task MDPs ###')
	mdps_a = { }
	for i in tqdm(range(len(goals)), desc='Single Task MDPs'):
		c = acmw.generate_costs_varied(goals[i], X_a, A_a, P_a)
		mdp = MDP(X_a, A_a, P_a, c, 0.9, get_goal_states(X_a, goals[i]))
		mdps_a['mdp' + str(i + 1)] = mdp
	print('Legible task MDP')
	task_mdp_a = LegibleTaskMDP(X_a, A_a, P_a, 0.9, goal, goals, list(mdps_a.values( )), 2.0,
								get_goal_states(X_a, goal))
	
	print('### Computing Optimal policy ###')
	time1 = time.time( )
	pol_a, Q_a = mdps_a['mdp' + str(goals.index(goal) + 1)].policy_iteration( )
	print('Took %.3f seconds to compute policy' % (time.time( ) - time1))
	
	print('### Computing Legible policy ###')
	time1 = time.time( )
	task_pol_a, task_Q_a = task_mdp_a.policy_iteration( )
	print('Took %.3f seconds to compute policy' % (time.time( ) - time1))
	
	print('#######################################')
	print('#####   Limit Collect Maze World  #####')
	print('#######################################')
	print('### Generating World ###')
	cmw = LimitedCollectMazeWorld( )
	X_l, A_l, P_l = cmw.generate_world(n_rows, n_cols, objs_states)
	
	print('### Computing Costs and Creating Task MDPs ###')
	mdps_l = { }
	for i in range(len(goals)):
		c = acmw.generate_costs_varied(goals[i], X_l, A_l, P_l)
		mdp = MDP(X_l, A_l, P_l, c, 0.9, get_goal_states(X_l, goals[i]))
		mdps_l['mdp' + str(i + 1)] = mdp
	task_mdp_l = LegibleTaskMDP(X_l, A_l, P_l, 0.9, goal, goals, list(mdps_l.values( )), 2.0,
								get_goal_states(X_l, goal))
	
	print('### Computing Optimal policy ###')
	time1 = time.time( )
	pol_l, Q1 = mdps_l['mdp' + str(goals.index(goal) + 1)].policy_iteration( )
	print('Took %.3f seconds to compute policy' % (time.time( ) - time1))
	
	print('### Computing Legible policy ###')
	time1 = time.time( )
	task_pol_l, task_Q = task_mdp_l.policy_iteration( )
	print('Took %.3f seconds to compute policy' % (time.time( ) - time1))
	
	print('######################################')
	print('############ TRAJECTORIES ############')
	print('######################################')
	print('#####   Auto Collect Maze World  #####')
	print('######################################')
	# print('Optimal trajectory for task: ' + goal)
	# t1, a1 = mdps_a[str(goals.index(goal) + 1)].trajectory(x0, pol_a)
	# print(t1)
	# print(a1)
	#
	# print('Legible trajectory for task: ' + goal)
	# task_traj, task_act = task_mdp_a.trajectory(x0, task_pol_a)
	# print(task_traj)
	# print(task_act)
	#
	print('Getting model performance!!')
	clock_1 = time.time( )
	mdp_r, mdp_rl, leg_mdp_r, leg_mdp_rl = simulate(mdps_a['mdp' + str(goals.index(goal) + 1)], pol_a,
													task_mdp_a, task_pol_a, x0, 10)
	time_simulation = time.time( ) - clock_1
	print('Simulation length = %.3f' % time_simulation)
	print('Optimal Policy performance:\nReward: %.3f\nLegible Reward: %.3f' % (mdp_r, mdp_rl))
	print('legible Policy performance:\nReward: %.3f\nLegible Reward: %.3f' % (leg_mdp_r, leg_mdp_rl))
	print('#######################################')
	print('#####   Limit Collect Maze World  #####')
	print('#######################################')
	# print('Optimal trajectory for task: ' + goal)
	# t1, a1 = mdps_l[str(goals.index(goal) + 1)].trajectory(x0, pol_l)
	# print(t1)
	# print(a1)
	#
	# print('Legible trajectory for task: ' + goal)
	# task_traj, task_act = task_mdp_l.trajectory(x0, task_pol_l)
	# print(task_traj)
	# print(task_act)
	print('Getting model performance!!')
	clock_1 = time.time( )
	mdp_r, mdp_rl, leg_mdp_r, leg_mdp_rl = simulate(mdps_l['mdp' + str(goals.index(goal) + 1)], pol_l,
													task_mdp_l, task_pol_l, x0, 10)
	time_simulation = time.time( ) - clock_1
	print('Simulation length = %.3f' % time_simulation)
	print('Optimal Policy performance:\nReward: %.3f\nLegible Reward: %.3f' % (mdp_r, mdp_rl))
	print('legible Policy performance:\nReward: %.3f\nLegible Reward: %.3f' % (leg_mdp_r, leg_mdp_rl))


if __name__ == '__main__':
	main( )