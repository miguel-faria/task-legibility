#!/usr/bin/env python
import sys
import os
import time
import signal
import json

import numpy as np
import argparse
import yaml
import csv
np.set_printoptions(precision=5, linewidth=1000)

from termcolor import colored
from tqdm import tqdm
from mdp import LegibleTaskMDP, MDP, LearnerMDP, Utilities
from mazeworld import AutoCollectMazeWord, WallAutoCollectMazeWorld, SimpleWallMazeWorld2
from utilities import store_savepoint, load_savepoint
from typing import Dict, List, Tuple
from pathlib import Path
from multiprocessing import Process


def write_iteration_results_csv(csv_file: Path, results: Dict, access_type: str, fields: List[str], iteration_states: List[str], n_iteration: int) -> None:
    try:
        with open(csv_file, access_type) as csvfile:
            field_names = ['iteration'] + sorted(fields)
            it_idx = str(n_iteration) + ' ' + ' '.join(iteration_states)
            row = [it_idx]
            for key, val in sorted(results.items()):
                inner_keys = list(results[key].keys())
                for inner_key in inner_keys:
                    row += [results[key][inner_key]]
                    
            if access_type != 'a':
                headers = ', '.join(field_names)
                np.savetxt(fname=csvfile, X=np.array([row], dtype=object), delimiter=',', header=headers, fmt='%s', comments='')
            else:
                np.savetxt(fname=csvfile, X=np.array([row], dtype=object), delimiter=',', fmt='%s', comments='')
    
    except IOError as e:
        print(colored("I/O error: " + str(e), color='red'))


def write_full_results_csv(csv_file: Path, results: Dict, access_type: str, fields: List[str], action_type: str) -> None:
    try:
        print('Results to write: ' + str(list(results.items())))
        with open(csv_file, access_type) as csvfile:
            row = [action_type]
            for key, val in sorted(results.items()):
                inner_keys = list(results[key].keys())
                for inner_key in inner_keys:
                    row += [results[key][inner_key]]
                
            if access_type != 'a':
                headers = ', '.join(sorted(fields))
                np.savetxt(fname=csvfile, X=np.array([row], dtype=object), delimiter=',', header=headers, fmt='%s', comments='')
            else:
                np.savetxt(fname=csvfile, X=np.array([row], dtype=object), delimiter=',', fmt='%s', comments='')
    
    except IOError as e:
        print(colored("I/O error: " + str(e), color='red'))


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    
    return wrapped


def get_goal_states(states: np.ndarray, goal: str) -> List[int]:
    state_lst = list(states)
    return [state_lst.index(state) for state in states if state.find(goal) != -1]


def eval_trajectory(states: np.ndarray, actions: List[str], traj_len: int, learner: LearnerMDP, mdps: Dict[str, MDP], pols: Dict[str, np.ndarray],
                    n_reps: int, conf: float, state_goals: List[str], tasks: List[str], data_dir: Path, action_type: str, world: str, agent: str,
                    mode: str, header: bool, log_file: Path) -> None:
    
    sys.stdout = open(log_file, 'w')
    sys.stderr = open(log_file, 'a')
    n_trajs = 1
    # Verify if a savepoint exists to restart from
    savepoint_file = data_dir / 'results' / ('irl_evaluation_' + world + '_' + agent + '_' + mode + '_' + action_type + '.save')
    if savepoint_file.exists():
        print('Restarting evaluation. Loading savepoint.')
        eval_results, eval_begin = load_savepoint(savepoint_file)
    else:
        print('Starting evaluation from beginning.')
        eval_results = dict()
        eval_results['correct'] = dict()
        eval_results['p_confidence'] = dict()
        eval_begin = 0
   
    for rep in range(eval_begin, n_reps):
        
        x0 = state_goals[rep]
        iteraction_res = dict()
        iteraction_res['correct'] = dict()
        iteraction_res['p_confidence'] = dict()
        iteraction_res['trajectory'] = dict()
        
        for goal in tasks:
            print('------------------------------------------------------------------------')
            print('Evaluation for task: %s' % goal)
            mdp = mdps['mdp_' + str(tasks.index(goal) + 1)]
            pol = pols['mdp_' + str(tasks.index(goal) + 1)]
            
            print('Creating trajectories')
            traj = []
            rng_gen = np.random.default_rng(2021)
            task_traj, task_act = mdp.trajectory_len(x0, pol, traj_len, rng_gen)
            for i in range(len(task_traj)):
                traj += [[list(states).index(task_traj[i]), list(actions).index(task_act[i])]]
    
            print('Testing inference of cost')
            if goal not in eval_results['correct']:
                eval_results['correct'][goal] = np.zeros(traj_len)
            if goal not in eval_results['p_confidence']:
                eval_results['p_confidence'][goal] = np.zeros(traj_len)
            learner_count, learner_confidence = learner.learner_eval(conf, [traj], traj_len, 1, tasks.index(goal))
            eval_results['correct'][goal] += learner_count / (n_trajs * n_reps)
            eval_results['p_confidence'][goal] += learner_confidence / n_reps
            iteraction_res['correct'][goal] = learner_count
            iteraction_res['p_confidence'][goal] = learner_confidence
            iteraction_res['trajectory'][goal] = task_traj.T

        print('Done %d%% of repetitions. (%d/%d)' % (round((rep + 1) * 100 / n_reps, 0), (rep+1), n_reps))
        print('Storing iteration data to file.')
        it_file = data_dir / 'results' / ('irl_evaluation_results_' + world + '_' + agent + '_' + mode + '_' + action_type + '.csv')
        results_keys = list(iteraction_res.keys())
        field_names = []
        for key in results_keys:
            inner_keys = list(iteraction_res[key].keys())
            for inner_key in inner_keys:
                field_names += [str(key) + '_' + str(inner_key)]
        
        if rep < 1:
            write_iteration_results_csv(it_file, iteraction_res, 'w', field_names, [x0], rep)
        else:
            write_iteration_results_csv(it_file, iteraction_res, 'a', field_names, [x0], rep)

    # Write evaluation results to file
    print('Finished evaluation for %s actions.\n' % action_type)
    csv_file = data_dir / 'results' / ('irl_evaluation_results_' + world + '_' + agent + '_' + mode + '.csv')
    results_keys = list(eval_results.keys())
    field_names = ['action_type']
    for key in results_keys:
        inner_keys = list(eval_results[key].keys())
        for inner_key in inner_keys:
            field_names += [str(key) + '_' + str(inner_key)]
        
    if header:
        write_full_results_csv(csv_file, eval_results, 'w', field_names, action_type)
    else:
        write_full_results_csv(csv_file, eval_results, 'a', field_names, action_type)


def eval_samples(states: List[str], actions: List[str], batch_size: int, learner: LearnerMDP, pols: Dict[str, np.ndarray], n_reps: int, conf: float,
                 state_goals: List[List[str]], tasks: List[str], data_dir: Path, action_type: str, world: str, agent: str, mode: str,
                 header: bool, log_file: Path) -> None:
    
    sys.stdout = open(log_file, 'w')
    sys.stderr = open(log_file, 'a')
    n_batches = 1
    # Verify if a savepoint exists to restart from
    savepoint_file = data_dir / 'results' / ('irl_evaluation_' + world + '_' + agent + '_' + mode + '_' + action_type + '.save')
    if savepoint_file.exists():
        print('Restarting evaluation. Loading savepoint.')
        eval_results, eval_begin = load_savepoint(savepoint_file)
    else:
        print('Starting evaluation from beginning.')
        eval_results = dict()
        eval_results['correct'] = dict()
        eval_results['p_confidence'] = dict()
        eval_begin = 0
    
    for rep in range(eval_begin, n_reps):

        sample_states = state_goals[rep]
        iteraction_res = dict()
        iteraction_res['correct'] = dict()
        iteraction_res['p_confidence'] = dict()
        iteraction_res['trajectory'] = dict()
        
        for goal in tasks:
            # print('Goal: ' + goal)
            pol = pols['mdp_' + str(tasks.index(goal) + 1)]
            
            samples_tmp = []
            samples_seq = []
            rng_gen = np.random.default_rng(2021)
            for i in range(batch_size):
                state = sample_states[i]
                state_idx = states.index(state)
                action = rng_gen.choice(len(actions), p=pol[state_idx, :])
                samples_seq += [state]
                samples_tmp += [[state_idx, action]]
            samples = [np.array(samples_tmp)]
            
            print('Testing inference of cost')
            if goal not in eval_results['correct']:
                eval_results['correct'][goal] = np.zeros(batch_size)
            if goal not in eval_results['p_confidence']:
                eval_results['p_confidence'][goal] = np.zeros(batch_size)
            learner_count, learner_confidence = learner.learner_eval(conf, samples, batch_size, 1, tasks.index(goal))
            print('Correct count: ' + str(learner_count))
            print('Confidence: ' + str(learner_confidence))
            eval_results['correct'][goal] += learner_count / (n_batches * n_reps)
            eval_results['p_confidence'][goal] += learner_confidence / n_reps
            iteraction_res['correct'][goal] = learner_count
            iteraction_res['p_confidence'][goal] = learner_confidence
            iteraction_res['trajectory'][goal] = np.array(samples_seq).T
            
        print('Done %d%% of repetitions. (%d/%d)' % (round((rep + 1) * 100 / n_reps, 0), (rep+1), n_reps))
        print('Storing iteration data to file.')
        it_file = data_dir / 'results' / ('irl_evaluation_results_' + world + '_' + agent + '_' + mode + '_' + action_type + '.csv')
        results_keys = list(iteraction_res.keys())
        field_names = []
        for key in results_keys:
            inner_keys = list(iteraction_res[key].keys())
            for inner_key in inner_keys:
                field_names += [str(key) + '_' + str(inner_key)]

        if rep < 1:
            write_iteration_results_csv(it_file, iteraction_res, 'w', field_names, sample_states, rep)
        else:
            write_iteration_results_csv(it_file, iteraction_res, 'a', field_names, sample_states, rep)

    # Write evaluation results to file
    print('Finished evaluation for %s actions.\n' % action_type)
    csv_file = data_dir / 'results' / ('irl_evaluation_results_' + world + '_' + agent + '_' + mode + '.csv')
    results_keys = list(eval_results.keys())
    field_names = ['action_type']
    for key in results_keys:
        inner_keys = list(eval_results[key].keys())
        for inner_key in inner_keys:
            field_names += [str(key) + '_' + str(inner_key)]
    if header:
        write_full_results_csv(csv_file, eval_results, 'w', field_names, action_type)
    else:
        write_full_results_csv(csv_file, eval_results, 'a', field_names, action_type)


def main():

    WORLD_CONFIGS = {1: '10x10_world', 2: '10x10_world_2', 3: '10x10_world_3', 4: '10x10_world_4'}

    parser = argparse.ArgumentParser(description='IRL task legibility in stochastic environment argument parser')
    parser.add_argument('--agent', dest='agent', type=str, required=True, choices=['optimal', 'legible'],
                        help='Type of agent to use as learner, either optimal or legible')
    parser.add_argument('--world', dest='world', type=int, required=True, help='World config ID')
    parser.add_argument('--leg_func', dest='leg_func', type=str, required=True, choices=['leg_optimal', 'leg_weight'],
                        help='Function to compute (state, action) legibility. Values accepted: \'leg_optimal\' '
                             'that computes the most legible optimal action for state and \'leg_weight\' '
                             'that computes the optimal legible action leveraged by the proximity to objectives.')
    parser.add_argument('--fail_prob', dest='fail_chance', type=float, required=True,
                        help='Probability of movement failing and staying in same place.')
    parser.add_argument('--mode', dest='mode', type=str, required=True, choices=['trajectory', 'sample'],
                        help='Task legibility performance mode: \'trajectory\' to test performance for trajectories'
                             'starting in one specific initial position (initial positions must be in the \'trajectory\' '
                             'field in the \'irl_test.yaml\' config file);'
                             '\'sample\' gives the learner random (state, action) pairs in the gridworld instead '
                             'of trajectories between a starting position and a goal (initial positions must be in the \'sample\' '
                             'field in the \'irl_test.yaml\' config file).')
    parser.add_argument('--reps', dest='reps', type=int, required=True,
                        help='Number of repetitions for the evaluation cycle to clear rounding errors.')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Discount factor for the MDPs')
    parser.add_argument('--no_verbose', dest='verbose', action='store_false',
                        help='Discount factor for the MDPs')
    parser.add_argument('--traj_len', dest='traj_len', type=int, help='For single mode, length of states in the trajectory for each evaluation iteration.')
    parser.add_argument('--batch_size', dest='batch_size', type=int, help='For the random mode, number of state samples for each evaluation iteration.')

    # Parsing input arguments
    args = parser.parse_args()
    agent = args.agent
    world_id = args.world
    mode = args.mode
    leg_func = args.leg_func
    n_reps = args.reps
    fail_chance = args.fail_chance
    verbose = args.verbose
    world = WORLD_CONFIGS[world_id]

    # Setup script output files and locations
    script_parent_dir = Path(__file__).parent.absolute().parent.absolute()
    data_dir = script_parent_dir / 'data'
    if not os.path.exists(script_parent_dir / 'logs'):
        os.mkdir(script_parent_dir / 'logs')
    log_dir = script_parent_dir / 'logs'
    log_file = log_dir / ('irl_evaluation_log_' + world + '_' + agent + '_' + mode + '.txt')
    sys.stdout = open(log_file, 'w+')
    sys.stderr = open(log_file, 'a')

    # Load world configuration
    with open(data_dir / 'configs' / (world + '.yaml')) as file:
        config_params = yaml.full_load(file)

        n_cols = config_params['n_cols']
        n_rows = config_params['n_rows']
        walls = config_params['walls']
        task_states = config_params['task_states']
        tasks = config_params['tasks']
    
    print('######################################')
    print('#####   Auto Collect Maze World  #####')
    print('######################################')
    print('### Generating World ###')
    sys.stdout.flush()
    sys.stderr.flush()
    wacmw = WallAutoCollectMazeWorld()
    X_w, A_w, P_w = wacmw.generate_world(n_rows, n_cols, task_states, walls, 'stochastic', fail_chance)
    print('### Computing Costs and Creating Task MDPs ###')
    mdps = {}
    pols = {}
    q_mdps_w = {}
    v_mdps_w = {}
    leg_mdps = {}
    leg_pols = {}
    costs = []
    dists = []
    print('Creating optimal task MDPs')
    sys.stdout.flush()
    sys.stderr.flush()
    for i in range(len(tasks)):
        c = wacmw.generate_rewards(tasks[i], X_w, A_w)
        costs += [c]
        mdp = MDP(X_w, A_w, P_w, c, 0.9, get_goal_states(X_w, tasks[i]), 'rewards', verbose)
        pol, q = mdp.policy_iteration()
        v = Utilities.v_from_q(q, pol)
        q_mdps_w[tasks[i]] = q
        v_mdps_w[tasks[i]] = v
        # dists += [mdp.policy_dist(pol)]
        pols['mdp_' + str(i + 1)] = pol
        mdps['mdp_' + str(i + 1)] = mdp
    print('Creating legible task MDPs')
    sys.stdout.flush()
    sys.stderr.flush()
    leg_costs = []
    dists = np.array(dists)
    for i in range(len(tasks)):
        mdp = LegibleTaskMDP(X_w, A_w, P_w, 0.9, verbose, tasks[i], task_states, tasks, 0.5, get_goal_states(X_w, tasks[i]), 1,
                             leg_func, q_mdps=q_mdps_w, v_mdps=v_mdps_w, dists=dists)
        leg_pol, _ = mdp.policy_iteration(i)
        leg_costs += [mdp.costs[i, :]]
        leg_mdps['mdp_' + str(i + 1)] = mdp
        leg_pols['mdp_' + str(i + 1)] = leg_pol

    print('Creating IRL Agent')
    sys.stdout.flush()
    sys.stderr.flush()
    if agent.find('optim') != -1:
        c = costs
        sign = 1
    elif agent.find('leg') != -1:
        c = leg_costs
        sign = 1
    else:
        print(colored('[ERROR] Invalid agent learner type, exiting program!', color='red'))
        return
    learner = LearnerMDP(X_w, A_w, P_w, 0.9, c, sign, verbose)
   
    # Load test parameters
    print('Load Testing parameters')
    with open(data_dir / 'configs' / 'irl_test.yaml') as file:
        state_goals = yaml.full_load(file)
    
    conf = 1.0
    if mode.find('trajectory') != -1:
       
        print('Starting trajectory IRL evaluation. Launching subprocesses')
        sys.stdout.flush()
        sys.stderr.flush()
        procs = []
        log_file = log_dir / ('irl_evaluation_log_' + world + '_' + agent + '_' + mode + '_' + 'optimal' + '.txt')
        opt_process = Process(target=eval_trajectory,
                    args=(X_w, A_w, args.traj_len, learner, mdps, pols, n_reps, conf, state_goals[world]['trajectory'],
                          tasks, data_dir, 'optimal', world, agent, mode, True, log_file))
        log_file = log_dir / ('irl_evaluation_log_' + world + '_' + agent + '_' + mode + '_' + 'legible' + '.txt')
        leg_process = Process(target=eval_trajectory,
                    args=(X_w, A_w, args.traj_len, learner, leg_mdps, leg_pols, n_reps, conf, state_goals[world]['trajectory'],
                          tasks, data_dir, 'legible', world, agent, mode, False, log_file))
    
        opt_process.start()
        procs.append(opt_process)
        leg_process.start()
        procs.append(leg_process)
    
        print('Trajectory IRL evaluation finished. Joining subprocesses')
        sys.stdout.flush()
        sys.stderr.flush()
        for p in procs:
            p.join()
            
    elif mode.find('sample') != -1:
       
        print('Starting sample IRL evaluation. Launching subprocesses')
        sys.stdout.flush()
        sys.stderr.flush()
        procs = []
        log_file = log_dir / ('irl_evaluation_log_' + world + '_' + agent + '_' + mode + '_' + 'optimal' + '.txt')
        opt_process = Process(target=eval_samples,
                    args=(list(X_w), A_w, args.batch_size, learner, pols, n_reps, conf, state_goals[world]['sample'], tasks, data_dir,
                          'optimal', world, agent, mode, True, log_file))
        log_file = log_dir / ('irl_evaluation_log_' + world + '_' + agent + '_' + mode + '_' + 'legible' + '.txt')
        leg_process = Process(target=eval_samples,
                    args=(list(X_w), A_w, args.batch_size, learner, leg_pols, n_reps, conf, state_goals[world]['sample'], tasks, data_dir,
                          'legible', world, agent, mode, False, log_file))

        opt_process.start()
        procs.append(opt_process)
        leg_process.start()
        procs.append(leg_process)

        print('Sample IRL evaluation finished. Joining subprocesses')
        sys.stdout.flush()
        sys.stderr.flush()
        for p in procs:
            p.join()

    else:
        print(colored('[ERROR] Invalid performance mode, exiting program!', color='red'))
        return


if __name__ == '__main__':
    main()
