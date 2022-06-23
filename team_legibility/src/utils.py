#! /usr/bin/env python

import numpy as np

from typing import List, Dict, Tuple
from scipy.sparse import csr_matrix


def Ppi(transitions: List[csr_matrix], actions: np.ndarray, pol: np.ndarray) -> np.ndarray:
	nA = len(actions)
	ppi = pol[:, 0, None] * transitions[0].toarray()
	for i in range(1, nA):
		ppi += pol[:, i, None] * transitions[i].toarray()
	
	return ppi


def evaluate_pol(mdp: Tuple, pol: np.ndarray):
	X = mdp[0]
	A = mdp[1]
	P = mdp[2]
	c = mdp[3]
	gamma = mdp[4]
	
	nS = len(X)
	
	# Cost and Probs averaged by policy
	cpi = (pol * c).sum(axis=1)
	ppi = Ppi(P, A, pol)
	
	# J = (I - gamma*P)^-1 * c
	J = np.linalg.inv(np.eye(nS) - gamma * ppi).dot(cpi)
	
	return J[:, None]


def policy_iteration(mdp: Tuple[np.ndarray, List[str], List[csr_matrix], np.ndarray, float],
					 init_pol: np.ndarray = None, init_q: np.ndarray = None) -> (np.ndarray, np.ndarray):
	
	X = mdp[0]
	A = mdp[1]
	P = mdp[2]
	c = mdp[3]
	gamma = mdp[4]
	nX = len(X)
	nA = len(A)
	
	# Initialize pol
	if init_pol is None:
		pol = np.ones((nX, nA)) / nA
	else:
		pol = init_pol
	
	# Initialize Q
	if init_q is None:
		Q = np.zeros((nX, nA))
	else:
		Q = init_q
	
	quit = False
	i = 0
	
	while not quit:
	
		# print('Iteration %d' % (i + 1), end='\r')
	
		J = evaluate_pol(mdp, pol)
	
		for act in range(nA):
			Q[:, act, None] = c[:, act, None] + gamma * P[act].toarray().dot(J)
	
		# print(Q)
		Qmin = Q.max(axis=1, keepdims=True)
		polnew = np.isclose(Q, Qmin, atol=1e-10, rtol=1e-10).astype(int)
		polnew = polnew / polnew.sum(axis=1, keepdims=True)
	
		quit = (pol == polnew).all()
		pol = polnew
		i += 1
	
	# print('N. iterations: ', i)
	
	return pol, Q
