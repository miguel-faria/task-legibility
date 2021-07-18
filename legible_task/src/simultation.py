#! /usr/bin/env python

import numpy as np
import time
import argparse
import yaml
import re
import seaborn as sns
import matplotlib.pyplot as plt
import math
np.set_printoptions(precision=5)

from tqdm import tqdm
from mdp import LegibleTaskMDP, MDP, LearnerMDP
from mazeworld import AutoCollectMazeWord, WallAutoCollectMazeWorld


def main():

    def get_goal_states(states, goal):

        state_lst = list(states)
        return [state_lst.index(state) for state in states if state.find(goal) != -1]

    def get_initial_states(states, task_locs):

        return [state for state in states if state.find('N') != -1] + \
               [str(e[0]) + ' ' + str(e[1]) + ' ' + e[2] for e in task_locs]

    def simulate(optimal_mdp, optimal_pol, legible_mdp, leg_pol, x0, n_trajs):

        mdp_trajs = []
        tasks_trajs = []

        for _ in tqdm(range(n_trajs), desc='Simulate Trajectories'):
            traj, acts = optimal_mdp.trajectory(x0, optimal_pol)
            traj_leg, acts_leg = legible_mdp.trajectory(x0, leg_pol)
            mdp_trajs += [[traj, acts]]
            tasks_trajs += [[traj_leg, acts_leg]]

        optimal_reward = optimal_mdp.trajectory_reward(mdp_trajs)
        optimal_legibility = legible_mdp.trajectory_reward(mdp_trajs)
        legible_cost = optimal_mdp.trajectory_reward(tasks_trajs)
        legible_legibility = legible_mdp.trajectory_reward(tasks_trajs)

        return optimal_reward, optimal_legibility, legible_cost, legible_legibility

    POS = 1
    NEG = -1
    WORLD_CONFIGS = {1: '8x8_world.yaml', 2: '10x10_world.yaml', 3: '8x8_world_2.yaml', 4: '10x10_world_2.yaml'}

    parser = argparse.ArgumentParser(description='Task legibility in stochastic environment argument parser')
    parser.add_argument('--n_trajs', dest='n_trajs', type=int, required=True,
                        help='Number of trajectories to generate for each objective')
    parser.add_argument('--world', dest='world', type=int, required=True, help='World config ID')
    parser.add_argument('--mode', dest='mode', type=str, required=True, choices=['single', 'all'],
                        help='Task legibility performance mode: \'single\' to test performance for one specific'
                             ' initial position (requires specifying initial position with field --begin);'
                             '\'all\' to test performance for all possible initial positions')
    parser.add_argument('--begin', dest='x0', type=str,
                        help='Initial state in the format \'x y\', where x, y are the state coords')
    parser.add_argument('--leg_func', dest='leg_func', type=str, required=True, choices=['leg_optimal', 'leg_weight'],
                        help='Function to compute (state, action) legibility. Values accepted: \'leg_optimal\' '
                             'that computes the most legible optimal action for state and \'leg_weight\' '
                             'that computes the optimal legible action leveraged by the proximity to objectives.')
    parser.add_argument('--reps', dest='reps', type=int, required=True,
                        help='Number of repetitions for the evaluation cycle to clear rounding errors.')
    parser.add_argument('--fail_prob', dest='fail_chance', type=float, required=True,
                        help='Probability of movement failing and staying in same place.')

    args = parser.parse_args()
    world = args.world
    with open('../data/configs/' + WORLD_CONFIGS[world]) as file:
        config_params = yaml.full_load(file)

        n_cols = config_params['n_cols']
        n_rows = config_params['n_rows']
        walls = config_params['walls']
        task_states = config_params['task_states']
        tasks = config_params['tasks']

    n_trajs = args.n_trajs
    mode = args.mode
    leg_func = args.leg_func
    n_reps = args.reps
    fail_chance = args.fail_chance
    n_tasks = len(tasks)

    print('######################################')
    print('#####   Auto Collect Maze World  #####')
    print('######################################')
    print('### Generating World ###')
    wacmw = WallAutoCollectMazeWorld()
    X_w, A_w, P_w = wacmw.generate_world(n_rows, n_cols, task_states, walls, 'stochastic', fail_chance)
    if mode == 'single':
        x0 = [args.x0 + ' N']
    else:
        x0 = get_initial_states(X_w, task_states)

    print('### Computing Costs and Creating Task MDPs ###')
    optimal_mdps = {}
    q_mdps_w = []
    legible_mdps = {}
    for i in tqdm(range(n_tasks), desc='Single Task MDPs'):
        c = wacmw.generate_costs_varied(tasks[i], X_w, A_w, P_w)
        mdp = MDP(X_w, A_w, P_w, c, 0.9, get_goal_states(X_w, tasks[i]))
        _, q = mdp.policy_iteration()
        q_mdps_w += [q]
        optimal_mdps['mdp_' + str(i + 1)] = mdp
    print('### Legible task MDPs ###')
    for i in tqdm(range(n_tasks), desc='Legible Task MDPs'):
        mdp = LegibleTaskMDP(X_w, A_w, P_w, 0.9, tasks[i], task_states, tasks, 2.0,
                             get_goal_states(X_w, tasks[i]), NEG, leg_func, q_mdps=q_mdps_w)
        legible_mdps['leg_mdp_' + str(i + 1)] = mdp

    optimal_cost = np.zeros((n_tasks, n_rows, n_cols))
    legible_cost = np.zeros((n_tasks, n_rows, n_cols))
    optimal_legibility = np.zeros((n_tasks, n_rows, n_cols))
    legible_legibility = np.zeros((n_tasks, n_rows, n_cols))

    for _ in tqdm(range(n_reps), desc='Repetitions'):

        for goal in tasks:
            print('Task goal: ' + goal)
            goal_idx = tasks.index(goal)

            for x in x0:
                print('Initial State: ' + x)
                state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", x, re.I)
                x_row = int(state_split.group(1))
                x_col = int(state_split.group(2))

                optimal_mdp = optimal_mdps['mdp_' + str(goal_idx + 1)]
                legible_mdp = legible_mdps['leg_mdp_' + str(goal_idx + 1)]

                print('Computing optimal and legible policies')
                optimal_pol, _ = optimal_mdp.policy_iteration()
                legible_pol, _ = legible_mdp.policy_iteration()

                print('Testing optimal and legible MDPs performance starting in %s to goal %s' % (x, goal))
                opt_r, opt_l, leg_r, leg_l = simulate(optimal_mdp, optimal_pol, legible_mdp, legible_pol, x, n_trajs)

                optimal_cost[goal_idx, x_row - 1, x_col - 1] += opt_r / n_reps
                optimal_legibility[goal_idx, x_row - 1, x_col - 1] += opt_l / n_reps
                legible_cost[goal_idx, x_row - 1, x_col - 1] += leg_r / n_reps
                legible_legibility[goal_idx, x_row - 1, x_col - 1] += leg_l / n_reps

            print('\n')

    for goal in tasks:

        goal_idx = tasks.index(goal)
        cost_min = math.floor(min(np.min(optimal_cost[goal_idx]), np.min(legible_cost[goal_idx])))
        cost_max = math.ceil(max(np.max(optimal_cost[goal_idx]), np.max(legible_cost[goal_idx])))
        leg_min = math.floor(min(np.min(optimal_legibility[goal_idx]), np.min(legible_legibility[goal_idx])))
        leg_max = math.ceil(max(np.max(optimal_legibility[goal_idx]), np.max(legible_legibility[goal_idx])))

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.suptitle('Goal ' + goal)
        ax1.set_title('Optimal Agent')
        ax2.set_title('Legible Agent')
        plt.setp(ax1, ylabel='Legibility')
        plt.setp(ax3, ylabel='Costs')
        sns.heatmap(optimal_cost[goal_idx], cmap='Reds', vmin=cost_min, vmax=cost_max, linewidths=0.3, ax=ax3)
        ax3.invert_yaxis()
        sns.heatmap(optimal_legibility[goal_idx], cmap='Reds', vmin=leg_min, vmax=leg_max, linewidths=0.3, ax=ax1)
        ax1.invert_yaxis()
        sns.heatmap(legible_cost[goal_idx], cmap='Reds', vmin=cost_min, vmax=cost_max, linewidths=0.3, ax=ax4)
        ax4.invert_yaxis()
        sns.heatmap(legible_legibility[goal_idx], cmap='Reds', vmin=leg_min, vmax=leg_max, linewidths=0.3, ax=ax2)
        ax2.invert_yaxis()
        fig.show()

    x = input()


if __name__ == '__main__':
    main()
