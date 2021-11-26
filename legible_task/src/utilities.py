#! /usr/bin/env python

import numpy as np
import json

from tqdm import tqdm
from typing import List, Dict, Tuple
from pathlib import Path


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


def store_savepoint(file_path: Path, results: Dict, iteration: int) -> None:
    # Create JSON with save data
    save_data = dict()
    save_data['results'] = results
    save_data['iteration'] = iteration
    
    # Save data to file
    with open(file_path, 'w') as json_file:
        json.dump(save_data, json_file)


def load_savepoint(file_path: Path) -> Tuple[Dict, int]:
    json_file = open(file_path, 'r')
    data = json.load(json_file)
    
    return data['results'], data['iteration']


def value_iteration(mdp: Tuple[np.ndarray, List[str], Dict[str, np.ndarray], np.ndarray, float]) -> np.ndarray:
    X = mdp[0]
    A = mdp[1]
    P = mdp[2]
    c = mdp[3]
    gamma = mdp[4]

    nS = len(X)
    nA = len(A)

    J = np.zeros(nS)
    err = 1
    i = 0

    while err > 1e-8:

        Q = []
        for act in range(nA):
            Q += [c[:, act] + gamma * P[A[act]].dot(J)]

        Jnew = np.min(Q, axis=0)
        err = np.linalg.norm(J - Jnew)
        J = Jnew

        i += 1

    return J[:, None]


def policy_iteration(mdp: Tuple[np.ndarray, List[str], Dict[str, np.ndarray], np.ndarray, float]) -> (np.ndarray, np.ndarray):
    X = mdp[0]
    A = mdp[1]
    P = mdp[2]
    c = mdp[3]
    gamma = mdp[4]
    nX = len(X)
    nA = len(A)

    # Initialize pol
    pol = np.ones((nX, nA)) / nA

    # Initialize Q
    Q = np.zeros((nX, nA))

    quit = False
    i = 0

    while not quit:

        print('Iteration %d' % (i + 1), end='\r')

        J = mdp.evaluate_pol(pol)

        for act in range(nA):
            Q[:, act, None] = c[:, act, None] + gamma * P[A[act]].dot(J)

        Qmin = Q.max(axis=1, keepdims=True)
        polnew = np.isclose(Q, Qmin, atol=1e-10, rtol=1e-10).astype(int)
        polnew = polnew / polnew.sum(axis=1, keepdims=True)

        quit = (pol == polnew).all()
        pol = polnew
        i += 1

    print('N. iterations: ', i)

    return pol, Q


def trajectory(mdp: Tuple[np.ndarray, List[str], Dict[str, np.ndarray], np.ndarray, float], x0: str,
               pol: np.ndarray, goal_states: List[int]) -> (np.ndarray, np.ndarray):
    X = mdp[0]
    A = mdp[1]
    P = mdp[2]

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

        stop = (x in goal_states)
        if stop:
            actions += [A[np.random.choice(nA, p=pol[x, :])]]

    return np.array(traj), np.array(actions)


def trajectory_reward(mdp: Tuple[np.ndarray, List[str], Dict[str, np.ndarray], np.ndarray, float], trajs: np.ndarray) -> float:
    r_avg = 0
    X = list(mdp[0])
    A = list(mdp[1])
    c = mdp[3]
    gamma = mdp[4]

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


