#! /usr/bin/env python

import numpy as np
import tensorflow as tf
import json
import csv
import pickle

from typing import List, Tuple, Dict
from scipy.sparse import csr_matrix
from pathlib import Path
from termcolor import colored


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
		
		
def store_savepoint(file_path: Path, results: Dict, iteration: int) -> None:
	# Create dictionary with save data
	save_data = dict()
	save_data['results'] = results
	save_data['iteration'] = iteration
	
	# Save data to file
	with open(file_path, 'wb') as pickle_file:
		pickle.dump(save_data, pickle_file)


def load_savepoint(file_path: Path) -> Tuple[Dict, int]:
	pickle_file = open(file_path, 'rb')
	data = pickle.load(pickle_file)
	
	return data['results'], data['iteration'] + 1


def Ppi(transitions: List[np.ndarray], nA: int, pol: np.ndarray) -> np.ndarray:
	ppi = pol[:, 0, None] * transitions[0]
	for i in range(1, nA):
		ppi += pol[:, i, None] * transitions[i]
	
	return ppi


def Ppi_gpu(transitions: List[tf.Tensor], nA: int, pol: tf.Tensor) -> tf.Tensor:
	ppi = pol[:, 0, None] * transitions[0]
	for i in range(1, nA):
		ppi += pol[:, i, None] * transitions[i]
	
	return ppi


def evaluate_pol(transitions: List[np.ndarray], rewards: np.ndarray, gamma: float, pol: np.ndarray, nX: int, nA: int) -> np.ndarray:
	
	# Cost and Probs averaged by policy
	cpi = (pol * rewards).sum(axis=1)
	ppi = Ppi(transitions, nA, pol)
	
	# J = (I - gamma*P)^-1 * c
	J = np.linalg.inv(np.eye(nX) - gamma * ppi).dot(cpi)
	
	return J[:, None]


def evaluate_pol_gpu(transitions: List[tf.Tensor], rewards: tf.Tensor, gamma: float, pol: tf.Tensor, nX: int, nA: int) -> tf.Tensor:
	# Cost and Probs averaged by policy
	cpi = tf.math.reduce_sum((pol * rewards), axis=1)
	ppi = Ppi_gpu(transitions, nA, pol)
	
	# J = (I - gamma*P)^-1 * c
	J = np.linalg.inv(np.eye(nX) - gamma * ppi).dot(cpi)
	
	return J[:, None]


def policy_iteration(mdp: Tuple[np.ndarray, List[str], List[csr_matrix], np.ndarray, float],
					 init_pol: np.ndarray = None, init_q: np.ndarray = None, verbose: bool = False) -> (np.ndarray, np.ndarray):
	
	X = mdp[0]
	A = mdp[1]
	P = []
	for act in range(len(A)):
		P += [mdp[2][act].toarray()]
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
		
		if verbose:
			print('Iteration %d' % (i + 1))
	
		J = evaluate_pol(P, c, gamma, pol, nX, nA)
	
		for act in range(nA):
			Q[:, act, None] = c[:, act, None] + gamma * P[act].dot(J)
	
		if verbose:
			for x in range(nX):
				print(X[x], str(Q[x, :]))
		Qmin = Q.max(axis=1, keepdims=True)
		polnew = np.isclose(Q, Qmin, atol=1e-10, rtol=1e-10).astype(int)
		polnew = polnew / polnew.sum(axis=1, keepdims=True)
	
		quit = (pol == polnew).all()
		pol = polnew
		i += 1
	
	if verbose:
		print('N. iterations: ', i)
	
	return pol, Q


def policy_iteration_gpu(mdp: Tuple[Tuple, Tuple, List[tf.SparseTensor], tf.Tensor, float],
						 init_pol: tf.Tensor = None, init_q: tf.Tensor = None) -> (np.ndarray, tf.Tensor):
	X = mdp[0]
	A = mdp[1]
	P = []
	for act in range(len(A)):
		P += [tf.sparse.to_dense(mdp[2][act])]
	c = mdp[3]
	gamma = mdp[4]
	nX = len(X)
	nA = len(A)
	
	# Initialize pol
	if init_pol is None:
		pol = tf.ones((nX, nA), dtype=tf.dtypes.float64) / nA
	else:
		pol = init_pol
	pol = tf.Variable(pol)
	
	# Initialize Q
	if init_q is None:
		Q = tf.zeros((nX, nA), dtype=tf.dtypes.float64)
	else:
		Q = init_q
	Q = tf.Variable(Q)
	
	quit = False
	i = 0
	
	while not quit:
		
		# print('Iteration %d' % (i + 1), end='\r')
		
		J = evaluate_pol_gpu(P, c, gamma, pol, nX, nA)
		
		for act in range(nA):
			Q = Q[:, act, None].assign(c[:, act, None] + gamma * tf.matmul(P[act], J))
		
		# print(Q)
		Qmin = tf.math.reduce_max(Q, axis=1, keepdims=True)
		polnew = tf.convert_to_tensor(np.isclose(Q.numpy(), Qmin.numpy(), atol=1e-10, rtol=1e-10).astype(int))
		polnew = polnew / tf.math.reduce_sum(polnew, axis=1, keepdims=True)
		
		quit = tf.math.reduce_all(tf.math.equal(pol, polnew))
		pol = polnew
		i += 1
	
	# print('N. iterations: ', i)
	
	return pol.numpy(), Q
	

def adjacent_locs(state_loc: Tuple, field_size: Tuple) -> List[Tuple]:
	
	state_row, state_col = state_loc
	field_rows, field_cols = field_size
	return list({(min(state_row + 1, field_rows - 1), state_col), (max(state_row - 1, 0), state_col),
				 (state_row, max(state_col - 1, 0)), (state_row, min(state_col + 1, field_cols - 1))})