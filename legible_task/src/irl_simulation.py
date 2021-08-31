#! /usr/bin/env python

import numpy as np
import time
import argparse
import yaml
np.set_printoptions(precision=5)

from termcolor import colored
from tqdm import tqdm
from mdp import LegibleTaskMDP, MDP, LearnerMDP, Utilities
from mazeworld import AutoCollectMazeWord, WallAutoCollectMazeWorld
from typing import List, Dict, Tuple


def main():
    def get_goal_states(states, goal, with_objs=True, goals=None):
        if with_objs:
            state_lst = list(states)
            return [state_lst.index(x) for x in states if x.find(goal) != -1]
        else:
            state_lst = list(states)
            for g in goals:
                if g[2].find(goal) != -1:
                    return [state_lst.index(str(g[0]) + ' ' + str(g[1]))]

    WORLD_CONFIGS = {1: '8x8_world.yaml', 2: '10x10_world.yaml', 3: '8x8_world_2.yaml', 4: '10x10_world_2.yaml'}

    parser = argparse.ArgumentParser(description='IRL task legibility in stochastic environment argument parser')
    parser.add_argument('--agent', dest='agent', type=str, required=True, choices=['optimal', 'legible'],
                        help='Type of agent to user, either optimal or legible')
    parser.add_argument('--world', dest='world', type=int, required=True, help='World config ID')
    parser.add_argument('--leg_func', dest='leg_func', type=str, required=True, choices=['leg_optimal', 'leg_weight'],
                        help='Function to compute (state, action) legibility. Values accepted: \'leg_optimal\' '
                             'that computes the most legible optimal action for state and \'leg_weight\' '
                             'that computes the optimal legible action leveraged by the proximity to objectives.')
    parser.add_argument('--fail_prob', dest='fail_chance', type=float, required=True,
                        help='Probability of movement failing and staying in same place.')
    parser.add_argument('--mode', dest='mode', type=str, required=True, choices=['single', 'random'],
                        help='Task legibility performance mode: \'single\' to test performance for one specific'
                             ' initial position (requires specifying initial position with field --begin);'
                             '\'random\' gives the learner random (state, action) pairs in the gridworld instead'
                             ' of trajectories between a starting position and a goal.')
    parser.add_argument('--reps', dest='reps', type=int, required=True,
                        help='Number of repetitions for the evaluation cycle to clear rounding errors.')
    parser.add_argument('--n_trajs', dest='n_trajs', type=int, help='For single mode, number of trajectories to generate for each objective')
    parser.add_argument('--traj_len', dest='traj_len', type=int, help='For single mode, length of steps in each trajectory.')
    parser.add_argument('--begin', dest='x0', type=str,
                        help='For single mode, initial state in the format \'x y\', where x, y are the state coords')
    parser.add_argument('--n_samples', dest='n_samples', type=int, help='For the random mode, number of (state, action) samples')
    parser.add_argument('--batch_size', dest='batch_size', type=int, help='For the random mode, number of samples to test at once')

    args = parser.parse_args()
    agent = args.agent
    world = args.world
    mode = args.mode
    leg_func = args.leg_func
    n_reps = args.reps
    fail_chance = args.fail_chance

    with open('../data/configs/' + WORLD_CONFIGS[world]) as file:
        config_params = yaml.full_load(file)

        n_cols = config_params['n_cols']
        n_rows = config_params['n_rows']
        walls = config_params['walls']
        task_states = config_params['task_states']
        tasks = config_params['tasks']

    opt_correct_count = {}
    opt_average_confidence = {}
    leg_correct_count = {}
    leg_average_confidence = {}

    print('######################################')
    print('#####   Auto Collect Maze World  #####')
    print('######################################')
    print('### Generating World ###')
    wacmw = WallAutoCollectMazeWorld()
    X_w, A_w, P_w = wacmw.generate_world(n_rows, n_cols, task_states, walls, 'stochastic', fail_chance)
    print('### Computing Costs and Creating Task MDPs ###')
    mdps_w = {}
    mdp_pols = {}
    q_mdps_w = {}
    v_mdps_w = {}
    task_mdps_w = {}
    task_mdp_pols = {}
    costs = []
    dists = []
    print('Optimal task MDPs')
    for i in tqdm(range(len(tasks)), desc='Optimal Task MDPs'):
        c = wacmw.generate_rewards(tasks[i], X_w, A_w)
        costs += [c]
        mdp = MDP(X_w, A_w, P_w, c, 0.9, get_goal_states(X_w, tasks[i]), 'rewards')
        pol, q = mdp.policy_iteration()
        v = Utilities.v_from_q(q, pol)
        q_mdps_w[tasks[i]] = q
        v_mdps_w[tasks[i]] = v
        dists += [mdp.policy_dist(pol)]
        mdp_pols['mdp_' + str(i + 1)] = pol
        mdps_w['mdp_' + str(i + 1)] = mdp
    print('Legible task MDPs')
    leg_costs = []
    dists = np.array(dists)
    for i in tqdm(range(len(tasks)), desc='Legible Task MDPs'):
        mdp = LegibleTaskMDP(X_w, A_w, P_w, 0.9, tasks[i], task_states, tasks, 0.5, get_goal_states(X_w, tasks[i]), 1,
                             leg_func, q_mdps=q_mdps_w, v_mdps=v_mdps_w, dists=dists)
        leg_pol, _ = mdp.policy_iteration(i)
        leg_costs += [mdp.costs[i, :]]
        task_mdps_w['leg_mdp_' + str(i + 1)] = mdp
        task_mdp_pols['leg_mdp_' + str(i + 1)] = leg_pol

    print('Creating IRL Agent')
    if agent.find('optim') != -1:
        c = costs
        sign = -1
    elif agent.find('leg') != -1:
        c = leg_costs
        sign = 1
    else:
        print(colored('[ERROR] Invalid agent type, exiting program!', color='red'))
        return

    step = 1
    learner = LearnerMDP(X_w, A_w, P_w, 0.9, c, sign)

    for rep in range(n_reps):
        if mode.find('single') != -1:
            n_trajs = args.n_trajs
            traj_len = args.traj_len
            x0 = args.x0 + ' N'
            
            for goal in tasks:
    
                print('------------------------------------------------------------------------')
                print('Evaluation for task: %s' % goal)
                mdp = mdps_w['mdp_' + str(tasks.index(goal) + 1)]
                task_mdp = task_mdps_w['leg_mdp_' + str(tasks.index(goal) + 1)]
                opt_pol = mdp_pols['mdp_' + str(tasks.index(goal) + 1)]
                leg_pol = task_mdp_pols['leg_mdp_' + str(tasks.index(goal) + 1)]
    
                print('Creating demo trajectories')
                print('Optimal trajectories')
                opt_trajs = []
                for _ in tqdm(range(n_trajs)):
                    t1, a1 = mdp.trajectory_len(x0, opt_pol, traj_len)
                    traj = []
                    for i in range(len(t1)):
                        traj += [[list(X_w).index(t1[i]), list(A_w).index(a1[i])]]
                    opt_trajs += [np.array(traj)]
                
                print('Legible trajectories')
                leg_trajs = []
                for _ in tqdm(range(n_trajs)):
                    traj = []
                    task_traj, task_act = task_mdp.trajectory_len(x0, leg_pol, traj_len)
                    for i in range(len(task_traj)):
                        traj += [[list(X_w).index(task_traj[i]), list(A_w).index(task_act[i])]]
                    leg_trajs += [np.array(traj)]
    
                if goal not in opt_correct_count:
                    opt_correct_count[goal] = np.zeros(traj_len)
                if goal not in opt_average_confidence:
                    opt_average_confidence[goal] = np.zeros(traj_len)
                if goal not in leg_correct_count:
                    leg_correct_count[goal] = np.zeros(traj_len)
                if goal not in leg_average_confidence:
                    leg_average_confidence[goal] = np.zeros(traj_len)
    
                print('Testing inference of cost')
                opt_count, opt_confidence = learner.learner_eval(1, opt_trajs, traj_len, step, tasks.index(goal))
                leg_count, leg_confidence = learner.learner_eval(1, leg_trajs, traj_len, step, tasks.index(goal))
    
                opt_correct_count[goal] += opt_count / n_trajs
                leg_correct_count[goal] += leg_count / n_trajs
    
                opt_average_confidence[goal] += opt_confidence
                leg_average_confidence[goal] += leg_confidence
                
        elif mode.find('random') != -1:
            
            n_samples = args.n_samples
            batch_size = args.batch_size
            n_batches = n_samples // batch_size
            sample_states = []
            for i in range(n_batches):
                sample_states += [np.random.choice(len(X_w), size=batch_size)]
            sample_states = np.array(sample_states)
            
            for goal in tasks:
                opt_pol = mdp_pols['mdp_' + str(tasks.index(goal) + 1)]
                leg_pol = task_mdp_pols['leg_mdp_' + str(tasks.index(goal) + 1)]
                
                opt_samples = []
                leg_samples = []
                for i in range(n_batches):
                    opt_samples_tmp = []
                    leg_samples_tmp = []
                    for state in sample_states[i]:
                        opt_action = np.random.choice(len(A_w), p=opt_pol[state, :])
                        leg_action = np.random.choice(len(A_w), p=leg_pol[state, :])
                        
                        opt_samples_tmp += [[state, opt_action]]
                        leg_samples_tmp += [[state, leg_action]]
                    opt_samples += [np.array(opt_samples_tmp)]
                    leg_samples += [np.array(leg_samples_tmp)]
    
                if goal not in opt_correct_count:
                    opt_correct_count[goal] = np.zeros(batch_size)
                if goal not in opt_average_confidence:
                    opt_average_confidence[goal] = np.zeros(batch_size)
                if goal not in leg_correct_count:
                    leg_correct_count[goal] = np.zeros(batch_size)
                if goal not in leg_average_confidence:
                    leg_average_confidence[goal] = np.zeros(batch_size)
    
                print('Testing inference of cost')
                opt_count, opt_confidence = learner.learner_eval(1, opt_samples, batch_size, step, tasks.index(goal))
                leg_count, leg_confidence = learner.learner_eval(1, leg_samples, batch_size, step, tasks.index(goal))

                opt_correct_count[goal] += opt_count / n_batches
                leg_correct_count[goal] += leg_count / n_batches

                opt_average_confidence[goal] += opt_confidence
                leg_average_confidence[goal] += leg_confidence

        else:
            print(colored('[ERROR] Invalid performance mode, exiting program!', color='red'))
            return

        print('Done %d%% of repetitions' % (round(rep + 1 * 100 / n_reps, 0)))
        print('--------------------------------------------------')

    for goal in tasks:

        print('Performance results for agent %s and task %s' % (agent, goal))
        print('Optimal trajectory correct inference: ' + str(opt_correct_count[goal] / n_reps))
        print('Optimal trajectory inference confidence: ' + str(opt_average_confidence[goal] / n_reps))
        print('\n')
        print('Legible trajectory correct inference: ' + str(leg_correct_count[goal] / n_reps))
        print('Legible trajectory inference confidence: ' + str(leg_average_confidence[goal] / n_reps))
        print('\n')
        print('------------------------------------------------------------------------')


if __name__ == '__main__':
    main()
