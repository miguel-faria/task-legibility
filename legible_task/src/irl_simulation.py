#! /usr/bin/env python

import numpy as np
import time
import argparse
import yaml
np.set_printoptions(precision=5)

from termcolor import colored
from tqdm import tqdm
from mdp import LegibleTaskMDP, MDP, LearnerMDP
from mazeworld import AutoCollectMazeWord, WallAutoCollectMazeWorld


def main():

    def get_goal_states(states, goal):

        state_lst = list(states)
        return [state_lst.index(x) for x in states if x.find(goal) != -1]

    WORLD_CONFIGS = {1: '8x8_world.yaml', 2: '10x10_world.yaml'}

    parser = argparse.ArgumentParser(description='IRL task legibility argument parser')
    parser.add_argument('--agent', dest='agent', type=str, help='Type of agent to user, either optimal or legible')
    parser.add_argument('--n_trajs', dest='n_trajs', type=int,
                        help='Number of trajectories to generate for each objective')
    parser.add_argument('--world', dest='world', type=int, help='World config ID')
    parser.add_argument('--begin', dest='x0', type=str,
                        help='Initial state in the format \'x y\', where x, y are the state coords')

    args = parser.parse_args()
    agent = args.agent
    world = args.world
    n_trajs = args.n_trajs
    x0 = args.x0 + ' N'

    with open('../data/configs/' + WORLD_CONFIGS[world]) as file:
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
    wacmw = WallAutoCollectMazeWorld()
    X_w, A_w, P_w = wacmw.generate_world(n_rows, n_cols, task_states, walls)
    print('### Computing Costs and Creating Task MDPs ###')
    mdps_w = {}
    q_mdps_w = []
    task_mdps_w = {}
    costs = []
    for i in tqdm(range(len(tasks)), desc='Single Task MDPs'):
        c = wacmw.generate_costs_varied(tasks[i], X_w, A_w, P_w)
        costs += [c]
        mdp = MDP(X_w, A_w, P_w, c, 0.9, get_goal_states(X_w, tasks[i]))
        _, q = mdp.policy_iteration()
        q_mdps_w += [q]
        mdps_w['mdp' + str(i + 1)] = mdp
    print('Legible task MDP')
    leg_costs = []
    for i in tqdm(range(len(tasks)), desc='Legible Task MDPs'):
        mdp = LegibleTaskMDP(X_w, A_w, P_w, 0.9, tasks[i], tasks, 2.0, get_goal_states(X_w, tasks[i]),
                             q_mdps=q_mdps_w)
        leg_costs += [mdp.costs]
        task_mdps_w['leg_mdp_' + str(i + 1)] = mdp

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

    for goal in tasks:

        print('------------------------------------------------------------------------')
        print('Evaluation for task: %s' % goal)
        mdp = mdps_w['mdp' + str(tasks.index(goal) + 1)]
        task_mdp = task_mdps_w['leg_mdp_' + str(tasks.index(goal) + 1)]

        print('Computing Optimal policy')
        time1 = time.time()
        pol_w, Q_w = mdp.policy_iteration()
        print('Took %.3f seconds to compute policy' % (time.time() - time1))

        print('Computing Legible policy')
        time1 = time.time()
        task_pol_w, task_Q_w = task_mdp.policy_iteration()
        print('Took %.3f seconds to compute policy' % (time.time() - time1))

        print('Creating demo trajectories')
        print('Optimal trajectories')
        opt_trajs = []
        opt_traj_len = 0
        for _ in range(n_trajs):
            t1, a1 = mdps_w['mdp' + str(tasks.index(goal) + 1)].trajectory(x0, pol_w)
            opt_traj_len = max(opt_traj_len, len(t1))
            traj = []
            for i in range(len(t1)):
                traj += [[list(X_w).index(t1[i]), list(A_w).index(a1[i])]]
            opt_trajs += [np.array(traj)]
        print('Legible trajectories')
        leg_trajs = []
        leg_traj_len = 0
        for _ in range(n_trajs):
            traj = []
            task_traj, task_act = task_mdp.trajectory(x0, task_pol_w)
            leg_traj_len = max(len(task_traj), leg_traj_len)
            for i in range(len(task_traj)):
                traj += [[list(X_w).index(task_traj[i]), list(A_w).index(task_act[i])]]
            leg_trajs += [np.array(traj)]

        print('Testing inference of cost')
        opt_correct_count, opt_average_confidence = learner.learner_eval(1, opt_trajs, opt_traj_len, step,
                                                                         tasks.index(goal))
        leg_correct_count, leg_average_confidence = learner.learner_eval(1, leg_trajs, leg_traj_len, step,
                                                                         tasks.index(goal))

        print('Performance results for agent %s and task %s' % (agent, goal))
        print('Optimal trajectory correct inference: ' + str(opt_correct_count))
        print('Optimal trajectory inference confidence: ' + str(opt_average_confidence))
        print('\n')
        print('Legible trajectory correct inference: ' + str(leg_correct_count))
        print('Legible trajectory inference confidence: ' + str(leg_average_confidence))
        print('\n')
        print('------------------------------------------------------------------------')


if __name__ == '__main__':
    main()
