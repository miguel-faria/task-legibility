import numpy as np

from tqdm import tqdm


def value_iteration(mdp):
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


def policy_iteration(mdp):
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


def trajectory(mdp, x0, pol, goal_states):
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


def trajectory_reward(mdp, trajs):
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