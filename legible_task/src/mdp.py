#! /usr/bin/env python

import numpy as np
import time
import math
from tqdm import tqdm


class MDP(object):

    def __init__(self, x, a, p, c, gamma, goal_states):

        self._mdp = (x, a, p, c, gamma)
        self._goal_states = goal_states

    @property
    def states(self):
        return self._mdp[0]

    @property
    def actions(self):
        return self._mdp[1]

    @property
    def transitions_prob(self):
        return self._mdp[2]

    @property
    def costs(self):
        return self._mdp[3]

    @property
    def gamma(self):
        return self._mdp[4]

    @property
    def goals(self):
        return self._goal_states

    @property
    def mdp(self):
        return self._mdp

    @mdp.setter
    def mdp(self, x, a, p, c, gamma):
        self._mdp = (x, a, p, c, gamma)

    @goals.setter
    def goals(self, goals):
        self._goal_states = goals

    def evaluate_pol(self, pol):

        X = self._mdp[0]
        A = self._mdp[1]
        P = self._mdp[2]
        c = self._mdp[3]
        gamma = self._mdp[4]

        nS = len(X)
        nA = len(A)

        # Cost and Probs averaged by policy
        cpi = (pol * c).sum(axis=1)
        ppi = pol[:, 0, None] * P[A[0]]
        for i in range(1, nA):
            ppi += pol[:, i, None] * P[A[i]]

        # J = (I - gamma*P)^-1 * c
        J = np.linalg.inv(np.eye(nS) - gamma * ppi).dot(cpi)

        return J[:, None]

    def value_iteration(self):

        X = self._mdp[0]
        A = self._mdp[1]
        P = self._mdp[2]
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

            Jnew = np.min(Q, axis=0)
            err = np.linalg.norm(J - Jnew)
            J = Jnew

            i += 1

        return J[:, None]

    def policy_iteration(self):

        X = self._mdp[0]
        A = self._mdp[1]
        P = self._mdp[2]
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

            J = self.evaluate_pol(pol)

            for act in range(nA):
                Q[:, act, None] = c[:, act, None] + gamma * P[A[act]].dot(J)

            Qmin = Q.min(axis=1, keepdims=True)
            polnew = np.isclose(Q, Qmin, atol=1e-10, rtol=1e-10).astype(int)
            polnew = polnew / polnew.sum(axis=1, keepdims=True)

            quit = (pol == polnew).all()
            pol = polnew
            i += 1

        print('N. iterations: ', i)

        return pol, Q

    def trajectory_len(self, x0, pol, traj_len):

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

    def trajectory(self, x0, pol):

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

            stop = (x in self._goal_states)
            if stop:
                actions += [A[np.random.choice(nA, p=pol[x, :])]]

        return np.array(traj), np.array(actions)

    def all_trajectories(self, x0, pol):

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
            while not stop_inner:
                pol_act = np.nonzero(pol[x, :])[0]
                if len(pol_act) > 1:
                    for j in range(1, len(pol_act)):
                        x_tmp = np.random.choice(nX, p=P[A[pol_act[j]]][x, :])
                        started_trajs += [[list(traj) + [X[x_tmp]], list(a_traj) + [A[pol_act[j]]]]]

                a = pol_act[0]
                x = np.random.choice(nX, p=P[A[a]][x, :])

                traj += [X[x]]
                a_traj += [A[a]]

                stop_inner = (x in self._goal_states)
                if stop_inner:
                    a_traj += [A[np.random.choice(nA, p=pol[x, :])]]

            i += 1
            stop = (i >= len(started_trajs))
            trajs += [np.array(traj)]
            acts += [np.array(a_traj)]

        return np.array(trajs, dtype=object), np.array(acts, dtype=object)

    def trajectory_reward(self, trajs):

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
    
    def __init__(self, x, a, p, gamma, task, tasks, beta, goal_states, task_mdps=None, q_mdps=None):

        self._task = task
        self._tasks = tasks
        self._tasks_q = []
        if task_mdps:
            for mdp in task_mdps:
                _, qi = mdp.policy_iteration()
                self._tasks_q += [qi]
        elif q_mdps:
            self._tasks_q = q_mdps
        else:
            return
        self._beta = beta
        nX = len(x)
        nA = len(a)
        c = np.zeros((nX, nA))
        for i in range(nX):
            for j in range(nA):
                c[i, j] = self.cost(i, j)

        super().__init__(x, a, p, c, gamma, goal_states)

    def cost(self, x, a):
        
        task_idx = self._tasks.index(self._task)
        
        task_cost = np.exp(-self._beta * self._tasks_q[task_idx][x, a])
        tasks_sum = task_cost
        for i in range(len(self._tasks_q)):
            if i != task_idx:
                tasks_sum += np.exp(-self._beta * self._tasks_q[i][x, a])

        return task_cost / tasks_sum

    def value_iteration(self):

        X = self._mdp[0]
        A = self._mdp[1]
        P = self._mdp[2]
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

            Jnew = np.max(Q, axis=0)
            err = np.linalg.norm(J - Jnew)
            J = Jnew

            i += 1

        return J[:, None]

    def policy_iteration(self):

        X = self._mdp[0]
        A = self._mdp[1]
        P = self._mdp[2]
        c = self._mdp[3]
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

            print('Iteration %d' % (i + 1), end='\r')

            J = self.evaluate_pol(pol)

            for act in range(nA):
                Q[:, act, None] = c[:, act, None] + gamma * P[A[act]].dot(J)

            Qmax = Q.max(axis=1, keepdims=True)
            polnew = np.isclose(Q, Qmax, atol=1e-10, rtol=1e-10).astype(int)
            polnew = polnew / polnew.sum(axis=1, keepdims=True)

            quit = (pol == polnew).all()
            pol = polnew
            i += 1

        print('N. iterations: ', i)

        return pol, Q


class LearnerMDP(object):

    def __init__(self, x, a, p, gamma, rewards, sign):

        self._mdp_r = (x, a, p, gamma)
        self._pol = np.zeros((len(x), len(a)))
        self._reward = np.zeros((len(x), len(a)))
        self._reward_library = rewards
        self._sign = sign
        pol_library = []
        for i in range(len(rewards)):
            pol, _ = self.policy_iteration(rewards[i])
            pol_library += [pol]
        self._pol_library = np.array(pol_library)

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
        if not c.any():
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
            likelihoods += [self.likelihood(state, action, conf)]

        r_likelihood = np.cumprod(np.array(likelihoods), axis=0)[-1]
        max_likelihood = np.max(r_likelihood)
        low_magnitude = math.floor(math.log(np.min(r_likelihood), 10)) - 1
        p_max = np.isclose(r_likelihood, max_likelihood, atol=10**low_magnitude, rtol=10**low_magnitude).astype(int)
        p_max = p_max / p_max.sum()
        amax_likelihood = np.random.choice(len(self._reward_library), p=p_max)

        return self._reward_library[amax_likelihood], amax_likelihood

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
        if not c.any():
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

            quit = (pol == polnew).all()
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
        for i in range(demo_step, traj_len+1, demo_step):
            indexes += [i]

        if traj_len % demo_step == 0:
            n_idx = traj_len // demo_step
        else:
            n_idx = traj_len // demo_step + 1
            indexes += [traj_len]
        
        correct_count = np.zeros(n_idx)        
        for traj in tqdm(trajs):
            for i in tqdm(range(n_idx)):
                idx = indexes[i]
                reward, r_idx = self.birl_inference(traj[:idx], conf)
                if r_idx == goal:
                    correct_count[i] += 1

        return correct_count


def main():

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
    acmw = AutoCollectMazeWord()
    X_a, A_a, P_a = acmw.generate_world(n_rows, n_cols, objs_states)

    print('### Computing Costs and Creating Task MDPs ###')
    mdps_a = {}
    for i in tqdm(range(len(goals)), desc='Single Task MDPs'):
        c = acmw.generate_costs_varied(goals[i], X_a, A_a, P_a)
        mdp = MDP(X_a, A_a, P_a, c, 0.9, get_goal_states(X_a, goals[i]))
        mdps_a['mdp' + str(i + 1)] = mdp
    print('Legible task MDP')
    task_mdp_a = LegibleTaskMDP(X_a, A_a, P_a, 0.9, goal, goals, list(mdps_a.values()), 2.0,
                                get_goal_states(X_a, goal))

    print('### Computing Optimal policy ###')
    time1 = time.time()
    pol_a, Q_a = mdps_a['mdp' + str(goals.index(goal) + 1)].policy_iteration()
    print('Took %.3f seconds to compute policy' % (time.time() - time1))

    print('### Computing Legible policy ###')
    time1 = time.time()
    task_pol_a, task_Q_a = task_mdp_a.policy_iteration()
    print('Took %.3f seconds to compute policy' % (time.time() - time1))

    print('#######################################')
    print('#####   Limit Collect Maze World  #####')
    print('#######################################')
    print('### Generating World ###')
    cmw = LimitedCollectMazeWorld()
    X_l, A_l, P_l = cmw.generate_world(n_rows, n_cols, objs_states)

    print('### Computing Costs and Creating Task MDPs ###')
    mdps_l = {}
    for i in range(len(goals)):
        c = acmw.generate_costs_varied(goals[i], X_l, A_l, P_l)
        mdp = MDP(X_l, A_l, P_l, c, 0.9, get_goal_states(X_l, goals[i]))
        mdps_l['mdp' + str(i + 1)] = mdp
    task_mdp_l = LegibleTaskMDP(X_l, A_l, P_l, 0.9, goal, goals, list(mdps_l.values()), 2.0,
                                get_goal_states(X_l, goal))

    print('### Computing Optimal policy ###')
    time1 = time.time()
    pol_l, Q1 = mdps_l['mdp' + str(goals.index(goal) + 1)].policy_iteration()
    print('Took %.3f seconds to compute policy' % (time.time() - time1))

    print('### Computing Legible policy ###')
    time1 = time.time()
    task_pol_l, task_Q = task_mdp_l.policy_iteration()
    print('Took %.3f seconds to compute policy' % (time.time() - time1))

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
    clock_1 = time.time()
    mdp_r, mdp_rl, leg_mdp_r, leg_mdp_rl = simulate(mdps_a['mdp' + str(goals.index(goal) + 1)], pol_a,
                                                    task_mdp_a, task_pol_a, x0, 10)
    time_simulation = time.time() - clock_1
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
    clock_1 = time.time()
    mdp_r, mdp_rl, leg_mdp_r, leg_mdp_rl = simulate(mdps_l['mdp' + str(goals.index(goal) + 1)], pol_l,
                                                    task_mdp_l, task_pol_l, x0, 10)
    time_simulation = time.time() - clock_1
    print('Simulation length = %.3f' % time_simulation)
    print('Optimal Policy performance:\nReward: %.3f\nLegible Reward: %.3f' % (mdp_r, mdp_rl))
    print('legible Policy performance:\nReward: %.3f\nLegible Reward: %.3f' % (leg_mdp_r, leg_mdp_rl))


if __name__ == '__main__':
    main()

