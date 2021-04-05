#! /usr/bin/env python

import numpy as np
import time
np.set_printoptions(precision=5)

from tqdm import tqdm
from mdp import LegibleTaskMDP, MDP, LearnerMDP
from mazeworld import AutoCollectMazeWord, WallAutoCollectMazeWorld


def main():

    def get_goal_states(states, goal):

        state_lst = list(states)
        return [state_lst.index(x) for x in states if x.find(goal) != -1]

    POS = 1
    NEG = -1

    n_rows = 8
    n_cols = 8
    # objs_states = [(1, 4, 'P'), (3, 1, 'D'), (3, 3, 'C')]
    # objs_states = [(2, 1, 'P')]
    objs_states = [(6, 4, 'P'), (4, 7, 'D'), (5, 2, 'C'), (8, 1, 'L'), (6, 5, 'T'), (8, 8, 'O')]
    # objs_states = [(7, 2, 'P'), (4, 13, 'D'), (6, 9, 'C'), (2, 4, 'L'), (5, 1, 'T'), (9, 10, 'O'), (10, 15,'E'),
    #                (1, 14,'A')]
    walls = [[(0.5, x + 0.5) for x in range(0, n_rows + 1)],
             [(n_rows + 0.5, x + 0.5) for x in range(0, n_rows + 1)],
             [(x + 0.5, 0.5) for x in range(0, n_cols + 1)],
             [(x + 0.5, n_cols + 0.5) for x in range(0, n_cols + 1)],
             [(0.5, 2.5), (1.5, 2.5)],
             # [(2.5, 0.5), (2.5, 1.5)],
             [(2.5, x + 0.5) for x in range(1, 5)],
             [(2.5, x + 0.5) for x in range(5, 8)],
             [(x + 0.5, 4.5) for x in range(2, 7)],
             [(x + 0.5, 4.5) for x in range(7, 9)],
             [(5.5, x + 0.5) for x in range(4, 6)],
             [(5.5, x + 0.5) for x in range(6, 8)],
             [(x + 0.5, 7.5) for x in range(1, 5)],
             [(x + 0.5, 7.5) for x in range(5, 8)]]
    # x0 = np.random.choice([x for x in X_a if 'N' in x])
    x0 = '1 1 N'
    # goals = ['PDT', 'PTO', 'DLC', 'DLT', 'DTO', 'CTO', 'PLO', 'PCO', 'PDO']
    goals = ['P', 'D', 'C', 'L', 'T', 'O']
    # goals = ['P', 'D', 'C']
    # goal = 'PDO'
    goal = 'T'

    print('Initial State: ' + x0)
    print('######################################')
    print('#####   Auto Collect Maze World  #####')
    print('######################################')
    print('### Generating World ###')
    wacmw = WallAutoCollectMazeWorld()
    X_w, A_w, P_w = wacmw.generate_world(n_rows, n_cols, objs_states, walls)

    print('### Computing Costs and Creating Task MDPs ###')
    mdps_w = {}
    q_mdps_w = []
    task_mdps_w = {}
    costs = []
    for i in tqdm(range(len(goals)), desc='Single Task MDPs'):
        c = wacmw.generate_costs_varied(goals[i], X_w, A_w, P_w)
        costs += [c]
        mdp = MDP(X_w, A_w, P_w, c, 0.9, get_goal_states(X_w, goals[i]))
        _, q = mdp.policy_iteration()
        q_mdps_w += [q]
        mdps_w['mdp' + str(i + 1)] = mdp
    print('Legible task MDP')
    for i in tqdm(range(len(goals)), desc='Legible Task MDPs'):
        mdp = LegibleTaskMDP(X_w, A_w, P_w, 0.9, goals[i], goals, 2.0, get_goal_states(X_w, goals[i]),
                             # task_mdps=list(mdps_w.values()))
                             q_mdps=q_mdps_w)
        task_mdps_w['leg_mdp_' + str(i + 1)] = mdp
    task_mdp_w = task_mdps_w['leg_mdp_' + str(goals.index(goal) + 1)]

    print('### Computing Optimal policy ###')
    time1 = time.time()
    pol_w, Q_w = mdps_w['mdp' + str(goals.index(goal) + 1)].policy_iteration()
    print('Took %.3f seconds to compute policy' % (time.time() - time1))

    print('### Computing Legible policy ###')
    time1 = time.time()
    task_pol_w, task_Q_w = task_mdp_w.policy_iteration()
    print('Took %.3f seconds to compute policy' % (time.time() - time1))

    print('Optimal trajectory for task: ' + goal)
    t1, a1 = mdps_w['mdp' + str(goals.index(goal) + 1)].trajectory(x0, pol_w)
    print(t1)
    print(a1)

    print('Legible trajectory for task: ' + goal)
    task_traj, task_act = task_mdp_w.trajectory(x0, task_pol_w)
    print(task_traj)
    print(task_act)

    print('Create trajectories')
    opt_traj = []
    for i in range(len(t1)):
        opt_traj += [[list(X_w).index(t1[i]), list(A_w).index(a1[i])]]
    opt_traj = np.array(opt_traj)
    leg_traj = []
    for i in range(len(task_traj)):
        leg_traj += [[list(X_w).index(task_traj[i]), list(A_w).index(task_act[i])]]
    leg_traj = np.array(leg_traj)
    # trajs = []
    # for _ in range(25):
    #     x0 = np.random.choice([x for x in X_w if 'N' in x])
    #     x0 = '1 1 N'
    #     t1, a1 = mdps_w['mdp' + str(goals.index(goal) + 1)].trajectory(x0, pol_w)
    #     print(t1)
    #     traj = []
    #     for i in range(len(t1)):
    #         traj += [[list(X_w).index(t1[i]), list(A_w).index(a1[i])]]
    #
    #     trajs += [np.array(traj)]

    print('IRL Agent')
    opt_learner = LearnerMDP(X_w, A_w, P_w, 0.9, pol_w, costs, NEG)
    leg_learner = LearnerMDP(X_w, A_w, P_w, 0.9, task_pol_w, costs, POS)
    print('Optimal Trajectory')
    for i in range(2, len(opt_traj), 2):
        print('Demo size %i' % i)
        # opt_reward, opt_idx = opt_learner.birl_inference(opt_traj[:i], 0.9)
        leg_reward, leg_idx = leg_learner.birl_inference(opt_traj[:i], 0.9)
        print(leg_idx)
    print('Legible Trajectory')
    for i in range(2, len(opt_traj), 2):
        print('Demo size %i' % i)
        # opt_reward, opt_idx = opt_learner.birl_inference(leg_traj[:i], 0.9)
        leg_reward, leg_idx = leg_learner.birl_inference(leg_traj[:i], 0.9)
        print(leg_idx)

    # print('IRL Policy')
    # pol_irl_lw, q_irl_lw = learner.policy_iteration()

    # x0 = '1 1 N'
    # print('Optimal irl trajectory for task: ' + goal)
    # t1, a1 = learner.trajectory(goal, pol_irl_lw, x0)
    # print('Trajectory: ' + str(t1))
    # print('Cost: ' + str(mdps_w['mdp' + str(goals.index(goal) + 1)].trajectory_reward([[t1, a1]])))
    # print('Legible Reward: ' + str(task_mdp_w.trajectory_reward([[t1, a1]])))


if __name__ == '__main__':
    main()
