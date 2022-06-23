#! /usr/bin/env python

from __future__ import annotations
import numpy as np

from typing import List, Tuple, Dict
from scipy.sparse import csr_matrix


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
	def q_from_pol(mdp: MDPAgent, pol: np.ndarray, task_idx: int = None) -> np.ndarray:
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
			q[:, act, None] = c[:, act, None] + gamma * P[task_idx][A[act]].dot(J)
		
		return q
	
	@staticmethod
	def q_from_v(v: np.ndarray, mdp: MDPAgent, task_idx: int = None) -> np.ndarray:
		A = mdp.actions
		P = mdp.transitions_prob
		c = mdp.costs
		gamma = mdp.gamma
		
		nX = len(mdp.states)
		nA = len(A)
		q = np.zeros((nX, nA))
		
		for a_idx in range(nA):
			q[:, a_idx, None] = c[:, a_idx, None] + gamma * P[task_idx][A[a_idx]].dot(v)
		
		return q
	
	@staticmethod
	def v_from_pol(mdp: MDPAgent, pol: np.ndarray, feedback_type: str, task_idx: int = None) -> np.ndarray:
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
	def Ppi(transitions: List[csr_matrix], actions: Tuple, pol: np.ndarray) -> np.ndarray:
		nA = len(actions)
		ppi = pol[:, 0, None] * transitions[0].toarray()
		for i in range(1, nA):
			ppi += pol[:, i, None] * transitions[i].toarray()
		
		return ppi
	
	@staticmethod
	def Ppi_stack(transitions: Dict[str, np.ndarray], actions: Tuple, pol: np.ndarray) -> np.ndarray:
		nX = len(transitions[actions[0]])
		nA = len(actions)
		ppi = np.zeros((nX * nA, nX * nA))
		
		for a in range(nA):
			for a2 in range(nA):
				ppi[a * nX:(a + 1) * nX, a2 * nX:(a2 + 1) * nX] = transitions[actions[a]] * pol[:, a2, None]
		
		return ppi
	
	@staticmethod
	def Q_func_grad(transitions: Dict[str, np.ndarray], actions: Tuple, pol: np.ndarray, q_pi: np.ndarray, gamma: float) -> np.ndarray:
		nX, nA = q_pi.shape
		q_stack = q_pi.flatten('F')
		pi_stack = pol.flatten('F')
		ppi_stack = Utilities.Ppi_stack(transitions, actions, pol)
		
		q_grad = gamma * np.linalg.inv(np.eye(nX * nA) - gamma * ppi_stack).dot(ppi_stack) * np.diag(q_stack / pi_stack)
		
		return np.diagonal(q_grad).reshape((nX, nA), order='F')
	
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


class MDPAgent(object):
	
	"""
	Base autonomous MDP agent class
	"""
	def __init__(self, x: Tuple, a: Tuple, p: List[List[csr_matrix]], c: np.ndarray, gamma: float, feedback_type: str, verbose: bool):
		self._mdp = (x, a, p, c, gamma)
		self._verbose = verbose
		if feedback_type.lower().find('cost') != -1 or feedback_type.lower().find('reward') != -1:
			self._feedback_type = feedback_type
		else:
			print('Invalid feedback type, defaulting to use costs.')
			self._feedback_type = 'cost'
	
	@property
	def states(self) -> Tuple:
		return self._mdp[0]
	
	@property
	def actions(self) -> Tuple:
		return self._mdp[1]
	
	@property
	def transitions_prob(self) -> List[List[csr_matrix]]:
		return self._mdp[2]
	
	@property
	def costs(self) -> np.ndarray:
		return self._mdp[3]
	
	@property
	def gamma(self) -> float:
		return self._mdp[4]
	
	@property
	def mdp(self) -> Tuple[Tuple, Tuple, List[List[csr_matrix]], np.ndarray, float]:
		return self._mdp
	
	@property
	def verbose(self) -> bool:
		return self._verbose
	
	@mdp.setter
	def mdp(self, mdp: Tuple[np.ndarray, np.ndarray, List[List[csr_matrix]], np.ndarray, float]):
		self._mdp = mdp
	
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
	
	@staticmethod
	def get_possible_states(q: np.ndarray, goal_states: List[int]) -> np.ndarray:
		nonzerostates = np.nonzero(q.sum(axis=1))[0]
		possible_states = [np.delete(nonzerostates, np.argwhere(nonzerostates == g)) for g in goal_states][0]
		return possible_states
	
	def evaluate_pol(self, pol: np.ndarray, task_idx: int = None) -> np.ndarray:
		"""
		Compute cost-to-go for a given policy
		:param pol: a nX*nA numpy array with the agent's current policy
		:param task_idx: for multi objective scenarios, gives the index of the agent's objective to obtain correct cost or reward function
		:return: J: cost-to-go for current policy
		"""
		X = self._mdp[0]
		A = self._mdp[1]
		if task_idx is None:
			P = self._mdp[2][0]
			c = self._mdp[3]
		else:
			P = self._mdp[2][task_idx]
			c = self._mdp[3][task_idx]
		gamma = self._mdp[4]
		
		nX = len(X)
		
		# Cost and Probs averaged by policy
		cpi = (pol * c).sum(axis=1)
		ppi = Utilities.Ppi(P, A, pol)
		
		# J = (I - gamma*P)^-1 * c
		J = np.linalg.inv(np.eye(nX) - gamma * ppi).dot(cpi)
		
		return J[:, None]
	
	def value_iteration(self, task_idx: int = None) -> np.ndarray:
		"""
		Compute J* via value iteration
		:param task_idx: for multi objective scenarios, gives the index of the agent's objective to obtain correct cost or reward function
		:return: J*
		"""
		X = self._mdp[0]
		A = self._mdp[1]
		if task_idx is None:
			P = self._mdp[2][0]
			c = self._mdp[3]
		else:
			P = self._mdp[2][task_idx]
			c = self._mdp[3][task_idx]
		gamma = self._mdp[4]
		
		nX = len(X)
		nA = len(A)
		
		J = np.zeros(nX)
		err = 1
		i = 0
		
		while err > 1e-8:
			Q = []
			for act in range(nA):
				Q += [c[:, act] + gamma * P[act].toarray().dot(J)]
			
			if self._feedback_type.find('cost') != -1:
				Jnew = np.min(Q, axis=0)
			else:
				Jnew = np.max(Q, axis=0)
			err = np.linalg.norm(J - Jnew)
			J = Jnew
			
			i += 1
		
		return J[:, None]
	
	def policy_iteration(self, task_idx: int = None) -> (np.ndarray, np.ndarray):
		"""
		Compute the optimal policy via policy iteration
		:param task_idx: for multi objective scenarios, gives the index of the agent's objective to obtain correct cost or reward function
		:return: pol, Q: optimal policy and Q-values
		"""
		X = self._mdp[0]
		A = self._mdp[1]
		if task_idx is None:
			P = self._mdp[2][0]
			c = self._mdp[3]
		else:
			P = self._mdp[2][task_idx]
			c = self._mdp[3][task_idx]
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
				Q[:, act, None] = c[:, act, None] + gamma * P[act].toarray().dot(J)
			
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
