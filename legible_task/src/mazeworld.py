#! /usr/bin/env python
import itertools

import numpy as np
import re
from termcolor import colored
from abc import ABC
from itertools import combinations, permutations
from typing import List, Tuple, Dict
from src.mdpworld import MDPWorld


#############################
#### Vanilla Maze Worlds ####
#############################
class MazeWorld(MDPWorld):

    def generate_rewards(self, goal, states, actions):

        nX = len(states)
        nA = len(actions)

        c = np.zeros((nX, nA))

        for state in states:
            if state.find(goal) != -1:
                c[list(states).index(state), :] = 1.0

        return c

    def generate_costs(self, goal, states, actions):

        nX = len(states)
        nA = len(actions)

        c = np.ones((nX, nA))

        for state in states:
            if state.find(goal) != -1:
                c[list(states).index(state), :] = 0.0

        return c

    def generate_world(self, n_rows, n_cols, obj_states, prob_type, fail_chance=0.0):
        states = self.generate_states(n_rows, n_cols, obj_states)
        actions = self.generate_actions()
        if prob_type.lower().find('stoc') != -1:
            probabilities = self.generate_stochastic_probabilities(states, actions, obj_states, n_rows,
                                                                   n_cols, fail_chance)
        else:
            probabilities = self.generate_probabilities(states, actions, obj_states, n_rows, n_cols)

        return states, actions, probabilities


class AutoCollectMazeWord(MazeWorld):

    def generate_states(self, n_rows, n_cols, obj_states):

        states = []
        obj_loc = []
        objs = []
        for x, y, o in obj_states:
            obj_loc += [(x, y)]
            objs += [o]
        n_objs = len(objs)
        for i in range(n_rows):
            for j in range(n_cols):
                cur_loc = (i + 1, j + 1)
                if cur_loc not in obj_loc:
                    states += [''.join(str(x) + ' ' for x in cur_loc) + 'N']
                for n in range(1, n_objs + 1):
                    combs = combinations(objs, n)
                    for comb in combs:
                        if cur_loc in obj_loc and str(comb).find(objs[obj_loc.index(cur_loc)]) == -1:
                            continue
                        states += [''.join(str(x) + ' ' for x in cur_loc) + ''.join(comb)]

        return np.array(states)

    def generate_actions(self):
        return np.array(['U', 'D', 'L', 'R', 'N'])

    def generate_stochastic_probabilities(self, states, actions, obj_states, max_rows, max_cols, fail_chance):

        objs_loc = []
        objs = []
        for x, y, o in obj_states:
            objs_loc += [(x, y)]
            objs += [o]
        n_objs = len(objs)
        nX = len(states)
        P = {}
        state_lst = list(states)

        for a in actions:

            p = np.zeros((nX, nX))
            if a == 'U':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_row = min(max_rows, curr_row + 1)
                    next_loc = (tmp_nxt_row, curr_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj

                    if nxt_state in states:
                        nxt_idx = state_lst.index(nxt_state)

                    else:
                        obj_loc_idx = objs_loc.index(next_loc)
                        if curr_state_obj == 'N':
                            nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])

                        else:
                            nxt_obj_lst = list(curr_state_obj) + list(objs[obj_loc_idx])
                            obj_perm = [''.join(x) for x in list(permutations(nxt_obj_lst))]
                            nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[0]
                            for i in range(1, len(obj_perm)):
                                if nxt_state in states:
                                    break
                                nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[i]
                            nxt_idx = state_lst.index(nxt_state)

                    if state_idx != nxt_idx:
                        p[state_idx][nxt_idx] = 1.0 - fail_chance
                        p[state_idx][state_idx] = fail_chance

                    else:
                        p[state_idx][state_idx] = 1.0

            elif a == 'D':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_row = max(1, curr_row - 1)
                    next_loc = (tmp_nxt_row, curr_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj

                    if nxt_state in states:
                        nxt_idx = state_lst.index(nxt_state)

                    else:
                        obj_loc_idx = objs_loc.index(next_loc)
                        if curr_state_obj == 'N':
                            nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])

                        else:
                            nxt_obj_lst = list(curr_state_obj) + list(objs[obj_loc_idx])
                            obj_perm = [''.join(x) for x in list(permutations(nxt_obj_lst))]
                            nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[0]
                            for i in range(1, len(obj_perm)):
                                if nxt_state in states:
                                    break
                                nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[i]
                            nxt_idx = state_lst.index(nxt_state)

                    if state_idx != nxt_idx:
                        p[state_idx][nxt_idx] = 1.0 - fail_chance
                        p[state_idx][state_idx] = fail_chance

                    else:
                        p[state_idx][state_idx] = 1.0

            elif a == 'L':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_col = max(1, curr_col - 1)
                    next_loc = (curr_row, tmp_nxt_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj

                    if nxt_state in states:
                        nxt_idx = state_lst.index(nxt_state)

                    else:
                        obj_loc_idx = objs_loc.index(next_loc)
                        if curr_state_obj == 'N':
                            nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])

                        else:
                            nxt_obj_lst = list(curr_state_obj) + list(objs[obj_loc_idx])
                            obj_perm = [''.join(x) for x in list(permutations(nxt_obj_lst))]
                            nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[0]
                            for i in range(1, len(obj_perm)):
                                if nxt_state in states:
                                    break
                                nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[i]
                            nxt_idx = state_lst.index(nxt_state)

                    if state_idx != nxt_idx:
                        p[state_idx][nxt_idx] = 1.0 - fail_chance
                        p[state_idx][state_idx] = fail_chance

                    else:
                        p[state_idx][state_idx] = 1.0

            elif a == 'R':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_col = min(max_cols, curr_col + 1)
                    next_loc = (curr_row, tmp_nxt_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj
                    if nxt_state in states:
                        nxt_idx = state_lst.index(nxt_state)

                    else:
                        obj_loc_idx = objs_loc.index(next_loc)
                        if curr_state_obj == 'N':
                            nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])

                        else:
                            nxt_obj_lst = list(curr_state_obj) + list(objs[obj_loc_idx])
                            obj_perm = [''.join(x) for x in list(permutations(nxt_obj_lst))]
                            nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[0]
                            for i in range(1, len(obj_perm)):
                                if nxt_state in states:
                                    break
                                nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[i]
                            nxt_idx = state_lst.index(nxt_state)

                    if state_idx != nxt_idx:
                        p[state_idx][nxt_idx] = 1.0 - fail_chance
                        p[state_idx][state_idx] = fail_chance

                    else:
                        p[state_idx][state_idx] = 1.0

            elif a == 'N':
                p = np.eye(nX)

            else:
                print(colored('Action not recognized. Skipping matrix probability', 'red'))
                continue

            P[a] = p

        return P

    def generate_probabilities(self, states, actions, obj_states, max_rows, max_cols):

        objs_loc = []
        objs = []
        for x, y, o in obj_states:
            objs_loc += [(x, y)]
            objs += [o]
        n_objs = len(objs)
        nX = len(states)
        P = {}
        state_lst = list(states)

        for a in actions:

            p = np.zeros((nX, nX))
            if a == 'U':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_row = min(max_rows, curr_row + 1)
                    next_loc = (tmp_nxt_row, curr_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj

                    if nxt_state in states:
                        nxt_idx = state_lst.index(nxt_state)

                    else:
                        obj_loc_idx = objs_loc.index(next_loc)
                        if curr_state_obj == 'N':
                            nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])

                        else:
                            nxt_obj_lst = list(curr_state_obj) + list(objs[obj_loc_idx])
                            obj_perm = [''.join(x) for x in list(permutations(nxt_obj_lst))]
                            nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[0]
                            for i in range(1, len(obj_perm)):
                                if nxt_state in states:
                                    break
                                nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[i]
                            nxt_idx = state_lst.index(nxt_state)

                    p[state_idx][nxt_idx] = 1.0

            elif a == 'D':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_row = max(1, curr_row - 1)
                    next_loc = (tmp_nxt_row, curr_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj

                    if nxt_state in states:
                        nxt_idx = state_lst.index(nxt_state)

                    else:
                        obj_loc_idx = objs_loc.index(next_loc)
                        if curr_state_obj == 'N':
                            nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])

                        else:
                            nxt_obj_lst = list(curr_state_obj) + list(objs[obj_loc_idx])
                            obj_perm = [''.join(x) for x in list(permutations(nxt_obj_lst))]
                            nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[0]
                            for i in range(1, len(obj_perm)):
                                if nxt_state in states:
                                    break
                                nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[i]
                            nxt_idx = state_lst.index(nxt_state)

                    p[state_idx][nxt_idx] = 1.0

            elif a == 'L':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_col = max(1, curr_col - 1)
                    next_loc = (curr_row, tmp_nxt_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj
                    if nxt_state in states:
                        nxt_idx = state_lst.index(nxt_state)

                    else:
                        obj_loc_idx = objs_loc.index(next_loc)
                        if curr_state_obj == 'N':
                            nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])

                        else:
                            nxt_obj_lst = list(curr_state_obj) + list(objs[obj_loc_idx])
                            obj_perm = [''.join(x) for x in list(permutations(nxt_obj_lst))]
                            nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[0]
                            for i in range(1, len(obj_perm)):
                                if nxt_state in states:
                                    break
                                nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[i]
                            nxt_idx = state_lst.index(nxt_state)

                    p[state_idx][nxt_idx] = 1.0

            elif a == 'R':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_col = min(max_cols, curr_col + 1)
                    next_loc = (curr_row, tmp_nxt_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj
                    if nxt_state in states:
                        nxt_idx = state_lst.index(nxt_state)

                    else:
                        obj_loc_idx = objs_loc.index(next_loc)
                        if curr_state_obj == 'N':
                            nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])

                        else:
                            nxt_obj_lst = list(curr_state_obj) + list(objs[obj_loc_idx])
                            obj_perm = [''.join(x) for x in list(permutations(nxt_obj_lst))]
                            nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[0]
                            for i in range(1, len(obj_perm)):
                                if nxt_state in states:
                                    break
                                nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[i]
                            nxt_idx = state_lst.index(nxt_state)

                    p[state_idx][nxt_idx] = 1.0

            elif a == 'N':
                p = np.eye(nX)

            else:
                print(colored('Action not recognized. Skipping matrix probability', 'red'))
                continue

            P[a] = p

        return P

    def generate_costs_varied(self, goal, states, actions, probabilities):

        nX = len(states)
        nA = len(actions)

        c = np.ones((nX, nA)) * 0.8

        for state in states:
            state_idx = list(states).index(state)
            if state.find(goal) != -1:
                c[state_idx, :] = 0.0
            else:
                for a in actions:
                    action_idx = list(actions).index(a)
                    nxt_state = states[np.random.choice(nX, p=probabilities[a][state_idx, :])]
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    state_obj = state_split.group(3)
                    nxt_state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", nxt_state, re.I)
                    nxt_state_obj = nxt_state_split.group(3)
                    if state_obj != nxt_state_obj:
                        c[state_idx, action_idx] = 0.9

        return c


class LimitedCollectMazeWorld(MazeWorld):

    def generate_states(self, n_rows, n_cols, obj_states):

        states = []
        obj_loc = []
        objs = []
        for x, y, o in obj_states:
            obj_loc += [(x, y)]
            objs += [o]
        n_objs = len(objs)
        for i in range(n_rows):
            for j in range(n_cols):
                cur_loc = (i + 1, j + 1)
                states += [''.join(str(x) + ' ' for x in cur_loc) + 'N']
                for n in range(1, n_objs + 1):
                    combs = combinations(objs, n)
                    for comb in combs:
                        states += [''.join(str(x) + ' ' for x in cur_loc) + ''.join(comb)]

        return np.array(states)

    def generate_actions(self):
        return np.array(['U', 'D', 'L', 'R', 'G', 'P', 'N'])

    def generate_probabilities(self, states, actions, obj_states, max_rows, max_cols):

        objs_loc = []
        objs = []
        for x, y, o in obj_states:
            objs_loc += [(x, y)]
            objs += [o]
        nX = len(states)
        P = {}
        state_lst = list(states)

        for a in actions:

            p = np.zeros((nX, nX))

            if a == 'U':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_row = min(max_rows, curr_row + 1)
                    next_loc = (tmp_nxt_row, curr_col)
                    nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + curr_state_obj)
                    p[state_idx][nxt_idx] = 1.0

            elif a == 'D':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_row = max(1, curr_row - 1)
                    next_loc = (tmp_nxt_row, curr_col)
                    nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + curr_state_obj)
                    p[state_idx][nxt_idx] = 1.0

            elif a == 'L':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_col = max(1, curr_col - 1)
                    next_loc = (curr_row, tmp_nxt_col)
                    nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + curr_state_obj)
                    p[state_idx][nxt_idx] = 1.0

            elif a == 'R':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_col = min(max_cols, curr_col + 1)
                    next_loc = (curr_row, tmp_nxt_col)
                    nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + curr_state_obj)
                    p[state_idx][nxt_idx] = 1.0

            elif a == 'G':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    curr_loc = (curr_row, curr_col)
                    state_idx = state_lst.index(state)

                    if curr_loc in objs_loc:
                        obj_loc_idx = objs_loc.index(curr_loc)
                        if curr_state_obj == 'N':
                            nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in curr_loc) +
                                                      objs[obj_loc_idx])
                        elif len(curr_state_obj) < 3 and curr_state_obj.find(objs[obj_loc_idx]) == -1:
                            nxt_obj_lst = list(curr_state_obj) + list(objs[obj_loc_idx])
                            obj_perm = [''.join(x) for x in list(permutations(nxt_obj_lst))]
                            nxt_state = ''.join(str(x) + ' ' for x in curr_loc) + obj_perm[0]
                            for i in range(1, len(obj_perm)):
                                if nxt_state in states:
                                    break
                                nxt_state = ''.join(str(x) + ' ' for x in curr_loc) + obj_perm[i]
                            nxt_idx = state_lst.index(nxt_state)

                        else:
                            nxt_idx = state_idx
                    else:
                        nxt_idx = state_idx

                    p[state_idx][nxt_idx] = 1.0

            elif a == 'P':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    curr_loc = (curr_row, curr_col)
                    state_idx = state_lst.index(state)

                    if curr_loc in objs_loc:
                        obj_loc_idx = objs_loc.index(curr_loc)
                        if curr_state_obj == 'N':
                            nxt_idx = state_idx

                        elif len(curr_state_obj) > 1 and curr_state_obj.find(objs[obj_loc_idx]) != -1:
                            nxt_obj_lst = list(curr_state_obj)
                            nxt_obj_lst.remove(objs[obj_loc_idx])
                            nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in curr_loc) + ''.join(nxt_obj_lst))

                        else:
                            nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in curr_loc) + 'N')
                    else:
                        nxt_idx = state_idx

                    p[state_idx][nxt_idx] = 1.0

            elif a == 'N':
                p = np.eye(nX)

            else:
                print(colored('Action not recognized. Skipping matrix probability', 'red'))
                continue

            P[a] = p

        return P

    def generate_costs_varied(self, goal, states, actions, probabilities):

        nX = len(states)
        nA = len(actions)

        c = np.ones((nX, nA)) * 0.8

        for state in states:
            state_idx = list(states).index(state)
            if state.find(goal) != -1:
                c[state_idx, :] = 0.0
            else:
                for a in actions:
                    if a == 'G' or a == 'P':
                        action_idx = list(actions).index(a)
                        nxt_state = states[np.random.choice(nX, p=probabilities[a][state_idx, :])]
                        state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                        state_obj = state_split.group(3)
                        nxt_state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", nxt_state, re.I)
                        nxt_state_obj = nxt_state_split.group(3)
                        if state_obj != nxt_state_obj:
                            c[state_idx, action_idx] = 0.9

        return c


################################
#### Maze Worlds with Walls ####
################################
class WallMazeWorld(MDPWorld):

    def wall_exists(self, state, action, walls):

        state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
        state_row = int(state_split.group(1))
        state_col = int(state_split.group(2))

        for wall in walls:
            if action == 'U':
                wall_left = (state_row + 0.5, state_col - 0.5)
                wall_right = (state_row + 0.5, state_col + 0.5)
                if wall_left in wall and wall_right in wall:
                    return True

            elif action == 'D':
                wall_left = (state_row - 0.5, state_col - 0.5)
                wall_right = (state_row - 0.5, state_col + 0.5)
                if wall_left in wall and wall_right in wall:
                    return True

            elif action == 'L':
                wall_up = (state_row - 0.5, state_col - 0.5)
                wall_down = (state_row + 0.5, state_col - 0.5)
                if wall_up in wall and wall_down in wall:
                    return True

            else:
                wall_up = (state_row - 0.5, state_col + 0.5)
                wall_down = (state_row + 0.5, state_col + 0.5)
                if wall_up in wall and wall_down in wall:
                    return True

        return False

    def generate_actions(self):
        return np.array(['U', 'D', 'L', 'R', 'N'])

    def generate_rewards(self, goal, states, actions):

        nX = len(states)
        nA = len(actions)

        c = np.zeros((nX, nA))

        for state in states:
            if state.find(goal) != -1:
                c[list(states).index(state), :] = 1.0

        return c

    def generate_costs(self, goal, states, actions):

        nX = len(states)
        nA = len(actions)

        c = np.ones((nX, nA))

        for state in states:
            if state.find(goal) != -1:
                c[list(states).index(state), :] = 0.0

        return c

    def generate_world(self, n_rows: int = 10, n_cols: int = 10, obj_states: List[Tuple[int, int, str]] = None, fail_chance: float = 0.0,
                       prob_type: str = 'stoc', max_grab_objs: int = 10, walls: List = None):
        states = self.generate_states(n_rows, n_cols, obj_states, max_grab_objs)
        actions = self.generate_actions()
        if prob_type.lower().find('stoc') != -1:
            probabilities = self.generate_stochastic_probabilities(states, actions, obj_states, n_rows,
                                                                   n_cols, walls, fail_chance, max_grab_objs)
        else:
            probabilities = self.generate_probabilities(states, actions, obj_states, n_rows, n_cols, walls,
                                                        max_grab_objs)

        return states, actions, probabilities


class WallAutoCollectMazeWorld(WallMazeWorld):

    def generate_states(self, n_rows, n_cols, obj_states, max_grab_objs=3):

        states = []
        obj_loc = []
        objs = []
        max_objs = min(max_grab_objs, len(obj_states))
        for x, y, o in obj_states:
            obj_loc += [(x, y)]
            objs += [o]
        for i in range(n_rows):
            for j in range(n_cols):
                cur_loc = (i + 1, j + 1)
                if cur_loc not in obj_loc:
                    states += [''.join(str(x) + ' ' for x in cur_loc) + 'N']
                for n in range(1, max_objs + 1):
                    combs = combinations(objs, n)
                    for comb in combs:
                        if cur_loc in obj_loc and str(comb).find(objs[obj_loc.index(cur_loc)]) == -1:
                            continue
                        states += [''.join(str(x) + ' ' for x in cur_loc) + ''.join(comb)]

        return np.array(states)

    def generate_stochastic_probabilities(self, states, actions, obj_states, max_rows, max_cols, walls,
                                          fail_chance, max_grab_objs=3):

        objs_loc = []
        objs = []
        for x, y, o in obj_states:
            objs_loc += [(x, y)]
            objs += [o]
        nX = len(states)
        P = {}
        state_lst = list(states)

        for a in actions:

            p = np.zeros((nX, nX))
            if a == 'U':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_row = min(max_rows, curr_row + 1)
                    next_loc = (tmp_nxt_row, curr_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx

                    else:
                        if nxt_state in states:
                            nxt_idx = state_lst.index(nxt_state)

                        else:
                            obj_loc_idx = objs_loc.index(next_loc)
                            if curr_state_obj == 'N':
                                nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])

                            else:
                                nxt_obj_lst = list(curr_state_obj) + list(objs[obj_loc_idx])
                                obj_perm = [''.join(x) for x in list(permutations(nxt_obj_lst))]
                                if min([len(perm) for perm in obj_perm]) <= max_grab_objs:
                                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[0]
                                    for i in range(1, len(obj_perm)):
                                        if nxt_state in states or len(obj_perm[i]) > max_grab_objs:
                                            break
                                        nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[i]
                                    nxt_idx = state_lst.index(nxt_state)
                                else:
                                    nxt_idx = state_idx

                    if state_idx != nxt_idx:
                        p[state_idx][nxt_idx] = 1.0 - fail_chance
                        p[state_idx][state_idx] = fail_chance

                    else:
                        p[state_idx][state_idx] = 1.0

            elif a == 'D':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_row = max(1, curr_row - 1)
                    next_loc = (tmp_nxt_row, curr_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx

                    else:
                        if nxt_state in states:
                            nxt_idx = state_lst.index(nxt_state)

                        else:
                            obj_loc_idx = objs_loc.index(next_loc)
                            if curr_state_obj == 'N':
                                nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])

                            else:
                                nxt_obj_lst = list(curr_state_obj) + list(objs[obj_loc_idx])
                                obj_perm = [''.join(x) for x in list(permutations(nxt_obj_lst))]
                                if min([len(perm) for perm in obj_perm]) <= max_grab_objs:
                                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[0]
                                    for i in range(1, len(obj_perm)):
                                        if nxt_state in states or len(obj_perm[i]) > max_grab_objs:
                                            break
                                        nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[i]
                                    nxt_idx = state_lst.index(nxt_state)
                                else:
                                    nxt_idx = state_idx

                    if state_idx != nxt_idx:
                        p[state_idx][nxt_idx] = 1.0 - fail_chance
                        p[state_idx][state_idx] = fail_chance

                    else:
                        p[state_idx][state_idx] = 1.0

            elif a == 'L':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_col = max(1, curr_col - 1)
                    next_loc = (curr_row, tmp_nxt_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx

                    else:
                        if nxt_state in states:
                            nxt_idx = state_lst.index(nxt_state)

                        else:
                            obj_loc_idx = objs_loc.index(next_loc)
                            if curr_state_obj == 'N':
                                nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])

                            else:
                                nxt_obj_lst = list(curr_state_obj) + list(objs[obj_loc_idx])
                                obj_perm = [''.join(x) for x in list(permutations(nxt_obj_lst))]
                                if min([len(perm) for perm in obj_perm]) <= max_grab_objs:
                                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[0]
                                    for i in range(1, len(obj_perm)):
                                        if nxt_state in states or len(obj_perm[i]) > max_grab_objs:
                                            break
                                        nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[i]
                                    nxt_idx = state_lst.index(nxt_state)
                                else:
                                    nxt_idx = state_idx

                    if state_idx != nxt_idx:
                        p[state_idx][nxt_idx] = 1.0 - fail_chance
                        p[state_idx][state_idx] = fail_chance

                    else:
                        p[state_idx][state_idx] = 1.0

            elif a == 'R':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_col = min(max_cols, curr_col + 1)
                    next_loc = (curr_row, tmp_nxt_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx

                    else:
                        if nxt_state in states:
                            nxt_idx = state_lst.index(nxt_state)

                        else:
                            obj_loc_idx = objs_loc.index(next_loc)
                            if curr_state_obj == 'N':
                                nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])

                            else:
                                nxt_obj_lst = list(curr_state_obj) + list(objs[obj_loc_idx])
                                obj_perm = [''.join(x) for x in list(permutations(nxt_obj_lst))]
                                if min([len(perm) for perm in obj_perm]) <= max_grab_objs:
                                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[0]
                                    for i in range(1, len(obj_perm)):
                                        if nxt_state in states or len(obj_perm[i]) > max_grab_objs:
                                            break
                                        nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[i]
                                    nxt_idx = state_lst.index(nxt_state)
                                else:
                                    nxt_idx = state_idx

                    if state_idx != nxt_idx:
                        p[state_idx][nxt_idx] = 1.0 - fail_chance
                        p[state_idx][state_idx] = fail_chance

                    else:
                        p[state_idx][state_idx] = 1.0

            elif a == 'N':
                p = np.eye(nX)

            else:
                print(colored('Action not recognized. Skipping matrix probability', 'red'))
                continue

            P[a] = p

        return P

    def generate_probabilities(self, states, actions, obj_states, max_rows, max_cols, walls, max_grab_objs=3):

        objs_loc = []
        objs = []
        for x, y, o in obj_states:
            objs_loc += [(x, y)]
            objs += [o]
        nX = len(states)
        P = {}
        state_lst = list(states)

        for a in actions:

            p = np.zeros((nX, nX))
            if a == 'U':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_row = min(max_rows, curr_row + 1)
                    next_loc = (tmp_nxt_row, curr_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx

                    else:
                        if nxt_state in states:
                            nxt_idx = state_lst.index(nxt_state)

                        else:
                            obj_loc_idx = objs_loc.index(next_loc)
                            if curr_state_obj == 'N':
                                nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])

                            else:
                                nxt_obj_lst = list(curr_state_obj) + list(objs[obj_loc_idx])
                                obj_perm = [''.join(x) for x in list(permutations(nxt_obj_lst))]
                                if min([len(perm) for perm in obj_perm]) <= max_grab_objs:
                                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[0]
                                    for i in range(1, len(obj_perm)):
                                        if nxt_state in states or len(obj_perm[i]) > max_grab_objs:
                                            break
                                        nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[i]
                                    nxt_idx = state_lst.index(nxt_state)
                                else:
                                    nxt_idx = state_idx

                    p[state_idx][nxt_idx] = 1.0

            elif a == 'D':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_row = max(1, curr_row - 1)
                    next_loc = (tmp_nxt_row, curr_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx

                    else:
                        if nxt_state in states:
                            nxt_idx = state_lst.index(nxt_state)

                        else:
                            obj_loc_idx = objs_loc.index(next_loc)
                            if curr_state_obj == 'N':
                                nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])

                            else:
                                nxt_obj_lst = list(curr_state_obj) + list(objs[obj_loc_idx])
                                obj_perm = [''.join(x) for x in list(permutations(nxt_obj_lst))]
                                if min([len(perm) for perm in obj_perm]) <= max_grab_objs:
                                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[0]
                                    for i in range(1, len(obj_perm)):
                                        if nxt_state in states or len(obj_perm[i]) > max_grab_objs:
                                            break
                                        nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[i]
                                    nxt_idx = state_lst.index(nxt_state)
                                else:
                                    nxt_idx = state_idx

                    p[state_idx][nxt_idx] = 1.0

            elif a == 'L':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_col = max(1, curr_col - 1)
                    next_loc = (curr_row, tmp_nxt_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx

                    else:
                        if nxt_state in states:
                            nxt_idx = state_lst.index(nxt_state)

                        else:
                            obj_loc_idx = objs_loc.index(next_loc)
                            if curr_state_obj == 'N':
                                nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])

                            else:
                                nxt_obj_lst = list(curr_state_obj) + list(objs[obj_loc_idx])
                                obj_perm = [''.join(x) for x in list(permutations(nxt_obj_lst))]
                                if min([len(perm) for perm in obj_perm]) <= max_grab_objs:
                                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[0]
                                    for i in range(1, len(obj_perm)):
                                        if nxt_state in states or len(obj_perm[i]) > max_grab_objs:
                                            break
                                        nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[i]
                                    nxt_idx = state_lst.index(nxt_state)
                                else:
                                    nxt_idx = state_idx

                    p[state_idx][nxt_idx] = 1.0

            elif a == 'R':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_col = min(max_cols, curr_col + 1)
                    next_loc = (curr_row, tmp_nxt_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx

                    else:
                        if nxt_state in states:
                            nxt_idx = state_lst.index(nxt_state)

                        else:
                            obj_loc_idx = objs_loc.index(next_loc)
                            if curr_state_obj == 'N':
                                nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])

                            else:
                                nxt_obj_lst = list(curr_state_obj) + list(objs[obj_loc_idx])
                                obj_perm = [''.join(x) for x in list(permutations(nxt_obj_lst))]
                                if min([len(perm) for perm in obj_perm]) <= max_grab_objs:
                                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[0]
                                    for i in range(1, len(obj_perm)):
                                        if nxt_state in states or len(obj_perm[i]) > max_grab_objs:
                                            break
                                        nxt_state = ''.join(str(x) + ' ' for x in next_loc) + obj_perm[i]
                                    nxt_idx = state_lst.index(nxt_state)
                                else:
                                    nxt_idx = state_idx

                    p[state_idx][nxt_idx] = 1.0

            elif a == 'N':
                p = np.eye(nX)

            else:
                print(colored('Action not recognized. Skipping matrix probability', 'red'))
                continue

            P[a] = p

        return P

    def generate_costs_varied(self, goal, states, actions, probabilities):

        nX = len(states)
        nA = len(actions)

        c = np.ones((nX, nA)) * 0.8

        for state in states:
            state_idx = list(states).index(state)
            if state.find(goal) != -1:
                c[state_idx, :] = 0.0
            else:
                for a in actions:
                    action_idx = list(actions).index(a)
                    nxt_state = states[np.random.choice(nX, p=probabilities[a][state_idx, :])]
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    state_obj = state_split.group(3)
                    nxt_state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", nxt_state, re.I)
                    nxt_state_obj = nxt_state_split.group(3)
                    if state_obj != nxt_state_obj:
                        c[state_idx, action_idx] = 0.9

        return c


class LimitedCollectWallMazeWorld(WallMazeWorld):

    def generate_world(self, n_rows, n_cols, obj_states, walls, prob_type, fail_chance=0.0, max_grab_objs=10):
        states = self.generate_states(n_rows, n_cols, obj_states, max_grab_objs)
        actions = self.generate_actions()
        if prob_type.lower().find('stoc') != -1:
            probabilities = self.generate_stochastic_probabilities(states, actions, obj_states, n_rows,
                                                                   n_cols, walls, fail_chance, max_grab_objs)
        else:
            probabilities = self.generate_probabilities(states, actions, obj_states, n_rows, n_cols,
                                                        walls, max_grab_objs)

        return states, actions, probabilities

    def generate_states(self, n_rows, n_cols, obj_states, max_grab_objs=3):

        states = []
        obj_loc = []
        objs = []
        max_objs = min(max_grab_objs, len(obj_states))
        for x, y, o in obj_states:
            obj_loc += [(x, y)]
            objs += [o]
        for i in range(n_rows):
            for j in range(n_cols):
                cur_loc = (i + 1, j + 1)
                states += [''.join(str(x) + ' ' for x in cur_loc) + 'N']
                for n in range(1, max_objs + 1):
                    combs = combinations(objs, n)
                    for comb in combs:
                        states += [''.join(str(x) + ' ' for x in cur_loc) + ''.join(comb)]

        return np.array(states)

    def generate_actions(self):
        return np.array(['U', 'D', 'L', 'R', 'G', 'P', 'N'])

    def generate_stochastic_probabilities(self, states, actions, obj_states, max_rows, max_cols, walls,
                                          fail_chance, max_grab_objs=3):

        objs_loc = []
        objs = []
        for x, y, o in obj_states:
            objs_loc += [(x, y)]
            objs += [o]
        nX = len(states)
        P = {}
        state_lst = list(states)

        for a in actions:

            p = np.zeros((nX, nX))

            if a == 'U':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx
                    else:
                        tmp_nxt_row = min(max_rows, curr_row + 1)
                        next_loc = (tmp_nxt_row, curr_col)
                        nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + curr_state_obj)

                    if state_idx != nxt_idx:
                        p[state_idx][nxt_idx] = 1.0 - fail_chance
                        p[state_idx][state_idx] = fail_chance

                    else:
                        p[state_idx][state_idx] = 1.0

            elif a == 'D':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx
                    else:
                        tmp_nxt_row = max(1, curr_row - 1)
                        next_loc = (tmp_nxt_row, curr_col)
                        nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + curr_state_obj)

                    if state_idx != nxt_idx:
                        p[state_idx][nxt_idx] = 1.0 - fail_chance
                        p[state_idx][state_idx] = fail_chance

                    else:
                        p[state_idx][state_idx] = 1.0

            elif a == 'L':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx
                    else:
                        tmp_nxt_col = max(1, curr_col - 1)
                        next_loc = (curr_row, tmp_nxt_col)
                        nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + curr_state_obj)

                    if state_idx != nxt_idx:
                        p[state_idx][nxt_idx] = 1.0 - fail_chance
                        p[state_idx][state_idx] = fail_chance

                    else:
                        p[state_idx][state_idx] = 1.0

            elif a == 'R':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx

                    else:
                        tmp_nxt_col = min(max_cols, curr_col + 1)
                        next_loc = (curr_row, tmp_nxt_col)
                        nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + curr_state_obj)

                    if state_idx != nxt_idx:
                        p[state_idx][nxt_idx] = 1.0 - fail_chance
                        p[state_idx][state_idx] = fail_chance

                    else:
                        p[state_idx][state_idx] = 1.0

            elif a == 'G':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    curr_loc = (curr_row, curr_col)
                    state_idx = state_lst.index(state)

                    if curr_loc in objs_loc:
                        obj_loc_idx = objs_loc.index(curr_loc)
                        if curr_state_obj == 'N':
                            nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in curr_loc) +
                                                      objs[obj_loc_idx])
                        elif len(curr_state_obj) < max_grab_objs and curr_state_obj.find(objs[obj_loc_idx]) == -1:
                            nxt_obj_lst = list(curr_state_obj) + list(objs[obj_loc_idx])
                            obj_perm = [''.join(x) for x in list(permutations(nxt_obj_lst))]
                            nxt_state = ''.join(str(x) + ' ' for x in curr_loc) + obj_perm[0]
                            for i in range(1, len(obj_perm)):
                                if nxt_state in states:
                                    break
                                nxt_state = ''.join(str(x) + ' ' for x in curr_loc) + obj_perm[i]
                            nxt_idx = state_lst.index(nxt_state)

                        else:
                            nxt_idx = state_idx
                    else:
                        nxt_idx = state_idx

                    if state_idx != nxt_idx:
                        p[state_idx][nxt_idx] = 1.0 - fail_chance
                        p[state_idx][state_idx] = fail_chance

                    else:
                        p[state_idx][state_idx] = 1.0

            elif a == 'P':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    curr_loc = (curr_row, curr_col)
                    state_idx = state_lst.index(state)

                    if curr_loc in objs_loc:
                        obj_loc_idx = objs_loc.index(curr_loc)
                        if curr_state_obj == 'N':
                            nxt_idx = state_idx

                        elif len(curr_state_obj) > 1 and curr_state_obj.find(objs[obj_loc_idx]) != -1:
                            nxt_obj_lst = list(curr_state_obj)
                            nxt_obj_lst.remove(objs[obj_loc_idx])
                            nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in curr_loc) + ''.join(nxt_obj_lst))

                        else:
                            nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in curr_loc) + 'N')
                    else:
                        nxt_idx = state_idx

                    if state_idx != nxt_idx:
                        p[state_idx][nxt_idx] = 1.0 - fail_chance
                        p[state_idx][state_idx] = fail_chance

                    else:
                        p[state_idx][state_idx] = 1.0

            elif a == 'N':
                p = np.eye(nX)

            else:
                print(colored('Action not recognized. Skipping matrix probability', 'red'))
                continue

            P[a] = p

        return P

    def generate_probabilities(self, states, actions, obj_states, max_rows, max_cols, walls, max_grab_objs=3):

        objs_loc = []
        objs = []
        for x, y, o in obj_states:
            objs_loc += [(x, y)]
            objs += [o]
        nX = len(states)
        P = {}
        state_lst = list(states)

        for a in actions:

            p = np.zeros((nX, nX))

            if a == 'U':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx
                    else:
                        tmp_nxt_row = min(max_rows, curr_row + 1)
                        next_loc = (tmp_nxt_row, curr_col)
                        nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + curr_state_obj)

                    p[state_idx][nxt_idx] = 1.0

            elif a == 'D':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx
                    else:
                        tmp_nxt_row = max(1, curr_row - 1)
                        next_loc = (tmp_nxt_row, curr_col)
                        nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + curr_state_obj)

                    p[state_idx][nxt_idx] = 1.0

            elif a == 'L':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx
                    else:
                        tmp_nxt_col = max(1, curr_col - 1)
                        next_loc = (curr_row, tmp_nxt_col)
                        nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + curr_state_obj)

                    p[state_idx][nxt_idx] = 1.0

            elif a == 'R':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx

                    else:
                        tmp_nxt_col = min(max_cols, curr_col + 1)
                        next_loc = (curr_row, tmp_nxt_col)
                        nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + curr_state_obj)

                    p[state_idx][nxt_idx] = 1.0

            elif a == 'G':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    curr_loc = (curr_row, curr_col)
                    state_idx = state_lst.index(state)

                    if curr_loc in objs_loc:
                        obj_loc_idx = objs_loc.index(curr_loc)
                        if curr_state_obj == 'N':
                            nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in curr_loc) +
                                                      objs[obj_loc_idx])
                        elif len(curr_state_obj) < max_grab_objs and curr_state_obj.find(objs[obj_loc_idx]) == -1:
                            nxt_obj_lst = list(curr_state_obj) + list(objs[obj_loc_idx])
                            obj_perm = [''.join(x) for x in list(permutations(nxt_obj_lst))]
                            nxt_state = ''.join(str(x) + ' ' for x in curr_loc) + obj_perm[0]
                            for i in range(1, len(obj_perm)):
                                if nxt_state in states:
                                    break
                                nxt_state = ''.join(str(x) + ' ' for x in curr_loc) + obj_perm[i]
                            nxt_idx = state_lst.index(nxt_state)

                        else:
                            nxt_idx = state_idx
                    else:
                        nxt_idx = state_idx

                    p[state_idx][nxt_idx] = 1.0

            elif a == 'P':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    curr_loc = (curr_row, curr_col)
                    state_idx = state_lst.index(state)

                    if curr_loc in objs_loc:
                        obj_loc_idx = objs_loc.index(curr_loc)
                        if curr_state_obj == 'N':
                            nxt_idx = state_idx

                        elif len(curr_state_obj) > 1 and curr_state_obj.find(objs[obj_loc_idx]) != -1:
                            nxt_obj_lst = list(curr_state_obj)
                            nxt_obj_lst.remove(objs[obj_loc_idx])
                            nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in curr_loc) + ''.join(nxt_obj_lst))

                        else:
                            nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in curr_loc) + 'N')
                    else:
                        nxt_idx = state_idx

                    p[state_idx][nxt_idx] = 1.0

            elif a == 'N':
                p = np.eye(nX)

            else:
                print(colored('Action not recognized. Skipping matrix probability', 'red'))
                continue

            P[a] = p

        return P

    def generate_costs_varied(self, goal, states, actions, probabilities):

        nX = len(states)
        nA = len(actions)

        c = np.ones((nX, nA)) * 0.8

        for state in states:
            state_idx = list(states).index(state)
            if state.find(goal) != -1:
                c[state_idx, :] = 0.0
            else:
                for a in actions:
                    if a == 'G' or a == 'P':
                        action_idx = list(actions).index(a)
                        nxt_state = states[np.random.choice(nX, p=probabilities[a][state_idx, :])]
                        state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                        state_obj = state_split.group(3)
                        nxt_state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", nxt_state, re.I)
                        nxt_state_obj = nxt_state_split.group(3)
                        if state_obj != nxt_state_obj:
                            c[state_idx, action_idx] = 0.9

        return c


###########################################
#### Simplified Maze Worlds with Walls ####
###########################################
class SimpleWallMazeWorld(object):

    def wall_exists(self, state, action, walls):

        state_split = re.match(r"([0-9]+) ([0-9]+)", state, re.I)
        state_row = int(state_split.group(1))
        state_col = int(state_split.group(2))

        for wall in walls:
            if action == 'U':
                wall_left = (state_row + 0.5, state_col - 0.5)
                wall_right = (state_row + 0.5, state_col + 0.5)
                if wall_left in wall and wall_right in wall:
                    return True

            elif action == 'D':
                wall_left = (state_row - 0.5, state_col - 0.5)
                wall_right = (state_row - 0.5, state_col + 0.5)
                if wall_left in wall and wall_right in wall:
                    return True

            elif action == 'L':
                wall_up = (state_row - 0.5, state_col - 0.5)
                wall_down = (state_row + 0.5, state_col - 0.5)
                if wall_up in wall and wall_down in wall:
                    return True

            else:
                wall_up = (state_row - 0.5, state_col + 0.5)
                wall_down = (state_row + 0.5, state_col + 0.5)
                if wall_up in wall and wall_down in wall:
                    return True

        return False

    def generate_states(self, n_rows, n_cols):

        states = []
        for i in range(n_rows):
            for j in range(n_cols):
                cur_loc = (i + 1, j + 1)
                states += [''.join(str(x) + ' ' for x in cur_loc)[:-1]]

        return np.array(states)

    def generate_actions(self):
        return np.array(['U', 'D', 'L', 'R', 'N'])

    def generate_stochastic_probabilities(self, states, actions, max_rows, max_cols, walls, fail_chance):

        nX = len(states)
        P = {}
        state_lst = list(states)

        for a in actions:

            p = np.zeros((nX, nX))
            if a == 'U':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    state_idx = state_lst.index(state)
                    tmp_nxt_row = min(max_rows, curr_row + 1)
                    next_loc = (tmp_nxt_row, curr_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc)[:-1]

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx

                    else:
                        nxt_idx = state_lst.index(nxt_state)


                    if state_idx != nxt_idx:
                        p[state_idx][nxt_idx] = 1.0 - fail_chance
                        p[state_idx][state_idx] = fail_chance

                    else:
                        p[state_idx][state_idx] = 1.0

            elif a == 'D':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    state_idx = state_lst.index(state)
                    tmp_nxt_row = max(1, curr_row - 1)
                    next_loc = (tmp_nxt_row, curr_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc)[:-1]

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx

                    else:
                        nxt_idx = state_lst.index(nxt_state)

                    if state_idx != nxt_idx:
                        p[state_idx][nxt_idx] = 1.0 - fail_chance
                        p[state_idx][state_idx] = fail_chance

                    else:
                        p[state_idx][state_idx] = 1.0

            elif a == 'L':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    state_idx = state_lst.index(state)
                    tmp_nxt_col = max(1, curr_col - 1)
                    next_loc = (curr_row, tmp_nxt_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc)[:-1]

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx

                    else:
                        nxt_idx = state_lst.index(nxt_state)

                    if state_idx != nxt_idx:
                        p[state_idx][nxt_idx] = 1.0 - fail_chance
                        p[state_idx][state_idx] = fail_chance

                    else:
                        p[state_idx][state_idx] = 1.0

            elif a == 'R':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    state_idx = state_lst.index(state)
                    tmp_nxt_col = min(max_cols, curr_col + 1)
                    next_loc = (curr_row, tmp_nxt_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc)[:-1]

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx

                    else:
                        nxt_idx = state_lst.index(nxt_state)

                    if state_idx != nxt_idx:
                        p[state_idx][nxt_idx] = 1.0 - fail_chance
                        p[state_idx][state_idx] = fail_chance

                    else:
                        p[state_idx][state_idx] = 1.0

            elif a == 'N':
                p = np.eye(nX)

            else:
                print(colored('Action not recognized. Skipping matrix probability', 'red'))
                continue

            P[a] = p

        return P

    def generate_probabilities(self, states, actions, max_rows, max_cols, walls):

        nX = len(states)
        P = {}
        state_lst = list(states)

        for a in actions:

            p = np.zeros((nX, nX))
            if a == 'U':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    state_idx = state_lst.index(state)
                    tmp_nxt_row = min(max_rows, curr_row + 1)
                    next_loc = (tmp_nxt_row, curr_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc)[:-1]

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx

                    else:
                        nxt_idx = state_lst.index(nxt_state)

                    p[state_idx][nxt_idx] = 1.0

            elif a == 'D':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    state_idx = state_lst.index(state)
                    tmp_nxt_row = max(1, curr_row - 1)
                    next_loc = (tmp_nxt_row, curr_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc)[:-1]

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx

                    else:
                        nxt_idx = state_lst.index(nxt_state)

                    p[state_idx][nxt_idx] = 1.0

            elif a == 'L':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    state_idx = state_lst.index(state)
                    tmp_nxt_col = max(1, curr_col - 1)
                    next_loc = (curr_row, tmp_nxt_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc)[:-1]

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx

                    else:
                        nxt_idx = state_lst.index(nxt_state)

                    p[state_idx][nxt_idx] = 1.0

            elif a == 'R':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    state_idx = state_lst.index(state)
                    tmp_nxt_col = min(max_cols, curr_col + 1)
                    next_loc = (curr_row, tmp_nxt_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc)[:-1]

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx

                    else:
                        nxt_idx = state_lst.index(nxt_state)

                    p[state_idx][nxt_idx] = 1.0

            elif a == 'N':
                p = np.eye(nX)

            else:
                print(colored('Action not recognized. Skipping matrix probability', 'red'))
                continue

            P[a] = p

        return P

    def generate_rewards(self, goal, states, actions):

        nX = len(states)
        nA = len(actions)

        goal_state = ''
        for g in self._goals:
            if g[2].find(goal) != -1:
                goal_state = str(g[0]) + ' ' + str(g[1])

        c = np.zeros((nX, nA))
        c[list(states).index(goal_state), :] = 1.0

        return c

    def generate_costs(self, goal, states, actions):

        nX = len(states)
        nA = len(actions)

        goal_state = ''
        for g in self._goals:
            if g[2].find(goal) != -1:
                goal_state = str(g[0]) + ' ' + str(g[1])

        c = np.ones((nX, nA))
        c[list(states).index(goal_state), :] = 0.0

        return c

    def generate_costs_varied(self, goal, states, actions, probabilities, goals):

        state_lst = list(states)
        act_lst = list(actions)
        nX = len(states)
        nA = len(actions)

        goal_state = ''
        for g in goals:
            if g[2].find(goal) != -1:
                goal_state = str(g[0]) + ' ' + str(g[1])

        c = np.ones((nX, nA)) * 0.8

        for state in states:
            state_idx = state_lst.index(state)
            if state.find(goal_state) != -1:
                c[state_idx, :] = 0.0
            else:
                for a in actions:
                    action_idx = act_lst.index(a)
                    nxt_states_idx = np.nonzero(probabilities[a][state_idx, :])[0]
                    for nxt_state_idx in nxt_states_idx:
                        nxt_state = states[nxt_state_idx]
                        if nxt_state.find(goal_state) == -1:
                            c[state_idx, action_idx] = 0.9

        return c

    def generate_world(self, n_rows, n_cols, obj_states, walls, prob_type, fail_chance=0.0):

        self._goals = obj_states
        states = self.generate_states(n_rows, n_cols)
        actions = self.generate_actions()
        if prob_type.lower().find('stoc') != -1:
            probabilities = self.generate_stochastic_probabilities(states, actions, n_rows, n_cols, walls, fail_chance)
        else:
            probabilities = self.generate_probabilities(states, actions, n_rows, n_cols, walls)

        return states, actions, probabilities


class SimpleWallMazeWorld2(object):

    def wall_exists(self, state, action, walls):

        state_split = re.match(r"([0-9]+) ([0-9]+)", state, re.I)
        state_row = int(state_split.group(1))
        state_col = int(state_split.group(2))

        for wall in walls:
            if action == 'U':
                wall_left = (state_row + 0.5, state_col - 0.5)
                wall_right = (state_row + 0.5, state_col + 0.5)
                if wall_left in wall and wall_right in wall:
                    return True

            elif action == 'D':
                wall_left = (state_row - 0.5, state_col - 0.5)
                wall_right = (state_row - 0.5, state_col + 0.5)
                if wall_left in wall and wall_right in wall:
                    return True

            elif action == 'L':
                wall_up = (state_row - 0.5, state_col - 0.5)
                wall_down = (state_row + 0.5, state_col - 0.5)
                if wall_up in wall and wall_down in wall:
                    return True

            else:
                wall_up = (state_row - 0.5, state_col + 0.5)
                wall_down = (state_row + 0.5, state_col + 0.5)
                if wall_up in wall and wall_down in wall:
                    return True

        return False

    def generate_states(self, n_rows, n_cols):

        states = []
        obj_loc = []
        objs = []
        for obj_state in self._goals:
            obj_loc += [(obj_state[0], obj_state[1])]
            objs += [obj_state[2]]
        for i in range(n_rows):
            for j in range(n_cols):
                cur_loc = (i + 1, j + 1)
                if cur_loc in obj_loc:
                    states += [''.join(str(x) + ' ' for x in cur_loc) + objs[obj_loc.index(cur_loc)]]
                else:
                    states += [''.join(str(x) + ' ' for x in cur_loc) + 'N']

        return np.array(states)

    def generate_actions(self):
        return np.array(['U', 'D', 'L', 'R'])

    def generate_stochastic_probabilities(self, states, actions, max_rows, max_cols, walls, fail_chance):

        nX = len(states)
        state_lst = list(states)

        objs_loc = []
        objs = []
        for x, y, o in self._goals:
            objs_loc += [(x, y)]
            objs += [o]

        P = {}

        for a in actions:

            p = np.zeros((nX, nX))
            if a == 'U':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_row = min(max_rows, curr_row + 1)
                    next_loc = (tmp_nxt_row, curr_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx

                    else:
                        if nxt_state in states:
                            nxt_idx = state_lst.index(nxt_state)

                        else:
                            if curr_state_obj == 'N':
                                obj_loc_idx = objs_loc.index(next_loc)
                                nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])

                            else:
                                nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + 'N')

                    if state_idx != nxt_idx:
                        p[state_idx][nxt_idx] = 1.0 - fail_chance
                        p[state_idx][state_idx] = fail_chance

                    else:
                        p[state_idx][state_idx] = 1.0

            elif a == 'D':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_row = max(1, curr_row - 1)
                    next_loc = (tmp_nxt_row, curr_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx

                    else:
                        if nxt_state in states:
                            nxt_idx = state_lst.index(nxt_state)

                        else:
                            if curr_state_obj == 'N':
                                obj_loc_idx = objs_loc.index(next_loc)
                                nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])

                            else:
                                nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + 'N')

                    if state_idx != nxt_idx:
                        p[state_idx][nxt_idx] = 1.0 - fail_chance
                        p[state_idx][state_idx] = fail_chance

                    else:
                        p[state_idx][state_idx] = 1.0

            elif a == 'L':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_col = max(1, curr_col - 1)
                    next_loc = (curr_row, tmp_nxt_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx

                    else:
                        if nxt_state in states:
                            nxt_idx = state_lst.index(nxt_state)

                        else:
                            if curr_state_obj == 'N':
                                obj_loc_idx = objs_loc.index(next_loc)
                                nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])

                            else:
                                nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + 'N')

                    if state_idx != nxt_idx:
                        p[state_idx][nxt_idx] = 1.0 - fail_chance
                        p[state_idx][state_idx] = fail_chance

                    else:
                        p[state_idx][state_idx] = 1.0

            elif a == 'R':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_col = min(max_cols, curr_col + 1)
                    next_loc = (curr_row, tmp_nxt_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx

                    else:
                        if nxt_state in states:
                            nxt_idx = state_lst.index(nxt_state)

                        else:
                            if curr_state_obj == 'N':
                                obj_loc_idx = objs_loc.index(next_loc)
                                nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])

                            else:
                                nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + 'N')

                    if state_idx != nxt_idx:
                        p[state_idx][nxt_idx] = 1.0 - fail_chance
                        p[state_idx][state_idx] = fail_chance

                    else:
                        p[state_idx][state_idx] = 1.0

            elif a == 'N':
                p = np.eye(nX)

            else:
                print(colored('Action not recognized. Skipping matrix probability', 'red'))
                continue

            P[a] = p

        return P

    def generate_probabilities(self, states, actions, max_rows, max_cols, walls):

        nX = len(states)
        state_lst = list(states)

        objs_loc = []
        objs = []
        for x, y, o in self._goals:
            objs_loc += [(x, y)]
            objs += [o]

        P = {}

        for a in actions:

            p = np.zeros((nX, nX))
            if a == 'U':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_row = max(1, curr_row - 1)
                    next_loc = (tmp_nxt_row, curr_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx

                    else:
                        if nxt_state in states:
                            nxt_idx = state_lst.index(nxt_state)

                        else:
                            if curr_state_obj == 'N':
                                obj_loc_idx = objs_loc.index(next_loc)
                                nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])

                            else:
                                nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + 'N')

                    p[state_idx][nxt_idx] = 1.0

            elif a == 'D':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_row = min(max_rows, curr_row + 1)
                    next_loc = (tmp_nxt_row, curr_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx

                    else:
                        if nxt_state in states:
                            nxt_idx = state_lst.index(nxt_state)

                        else:
                            if curr_state_obj == 'N':
                                obj_loc_idx = objs_loc.index(next_loc)
                                nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])

                            else:
                                nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + 'N')

                    p[state_idx][nxt_idx] = 1.0

            elif a == 'L':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_col = max(1, curr_col - 1)
                    next_loc = (curr_row, tmp_nxt_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx

                    else:
                        if nxt_state in states:
                            nxt_idx = state_lst.index(nxt_state)

                        else:
                            if curr_state_obj == 'N':
                                obj_loc_idx = objs_loc.index(next_loc)
                                nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])

                            else:
                                nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + 'N')

                    p[state_idx][nxt_idx] = 1.0

            elif a == 'R':
                for state in states:
                    state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
                    curr_row = int(state_split.group(1))
                    curr_col = int(state_split.group(2))
                    curr_state_obj = state_split.group(3)
                    state_idx = state_lst.index(state)
                    tmp_nxt_col = min(max_cols, curr_col + 1)
                    next_loc = (curr_row, tmp_nxt_col)
                    nxt_state = ''.join(str(x) + ' ' for x in next_loc) + curr_state_obj

                    if self.wall_exists(state, a, walls):
                        nxt_idx = state_idx

                    else:
                        if nxt_state in states:
                            nxt_idx = state_lst.index(nxt_state)

                        else:
                            if curr_state_obj == 'N':
                                obj_loc_idx = objs_loc.index(next_loc)
                                nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + objs[obj_loc_idx])

                            else:
                                nxt_idx = state_lst.index(''.join(str(x) + ' ' for x in next_loc) + 'N')

                    p[state_idx][nxt_idx] = 1.0

            elif a == 'N':
                p = np.eye(nX)

            else:
                print(colored('Action not recognized. Skipping matrix probability', 'red'))
                continue

            P[a] = p

        return P

    def generate_rewards(self, goal, states, actions):

        nX = len(states)
        nA = len(actions)

        c = np.zeros((nX, nA))

        for state in states:
            if state.find(goal) != -1:
                c[list(states).index(state), :] = 1.0

        return c

    def generate_costs(self, goal, states, actions):

        nX = len(states)
        nA = len(actions)

        c = np.ones((nX, nA))

        for state in states:
            if state.find(goal) != -1:
                c[list(states).index(state), :] = 0.0

        return c

    def generate_costs_varied(self, goal, states, actions, probabilities, goals):

        state_lst = list(states)
        act_lst = list(actions)
        nX = len(states)
        nA = len(actions)

        goal_state = ''
        for g in goals:
            if g[2].find(goal) != -1:
                goal_state = str(g[0]) + ' ' + str(g[1])

        c = np.ones((nX, nA)) * 0.8

        for state in states:
            state_idx = state_lst.index(state)
            if state.find(goal_state) != -1:
                c[state_idx, :] = 0.0
            else:
                for a in actions:
                    action_idx = act_lst.index(a)
                    nxt_states_idx = np.nonzero(probabilities[a][state_idx, :])[0]
                    for nxt_state_idx in nxt_states_idx:
                        nxt_state = states[nxt_state_idx]
                        if nxt_state.find(goal_state) == -1:
                            c[state_idx, action_idx] = 0.9

        return c

    def generate_world(self, n_rows, n_cols, obj_states, walls, prob_type, fail_chance=0.0):

        self._goals = obj_states
        states = self.generate_states(n_rows, n_cols)
        actions = self.generate_actions()
        if prob_type.lower().find('stoc') != -1:
            probabilities = self.generate_stochastic_probabilities(states, actions, n_rows, n_cols, walls, fail_chance)
        else:
            probabilities = self.generate_probabilities(states, actions, n_rows, n_cols, walls)

        return states, actions, probabilities


###############################################
#### Multiple Robot Maze Worlds with Walls ####
###############################################
class MultipleRobotMazeWorld(MDPWorld):
    
    def __init__(self, n_rows: int, n_cols: int, objs: List[str], obj_states: List[Tuple], walls: List[Tuple],
                 fail_chance: float = 0.0, n_robots: int = 2):
        
        self._n_rows = n_rows
        self._n_cols = n_cols
        self._objs = objs
        self._obj_states = obj_states
        self._walls = walls
        self._fail_chance = fail_chance
        self._n_robots = n_robots
        
        self._states = []
        self._actions = []
    
    @staticmethod
    def get_state_str(state_tuple: Tuple) -> str:
        return ', '.join(' '.join(str(x) for x in elem) for elem in state_tuple)
    
    @staticmethod
    def get_state_tuple(state_str: str) -> Tuple:
        state = []
        state_split = state_str.split(', ')
        for elem in state_split:
            try:
                state += [tuple(map(int, elem.split(' ')))]
            except ValueError:
                state += [tuple(elem.split(' '))]
        
        return tuple(state)
    
    @staticmethod
    def get_action_tuple(action_str: str) -> Tuple[str]:
        return tuple(action_str.split(' '))
    
    def wall_exists(self, state: Tuple[int, int], action: str) -> bool:

        state_row = state[0]
        state_col = state[1]
        
        if action == 'U':
            up_move = (min(self._n_rows, state_row + 1), state_col)
            if up_move in self._walls:
                return True

        elif action == 'D':
            down_move = (max(0, state_row - 1), state_col)
            if down_move in self._walls:
                return True

        elif action == 'L':
            left_move = (state_row, max(0, state_col - 1))
            if left_move in self._walls:
                return True

        elif action == 'R':
            right_move = (state_row, min(self._n_cols, state_col + 1))
            if right_move in self._walls:
                return True
        
        else:
            return False
            
        return False

    def generate_states(self) -> np.ndarray:

        states = []
        locs = []
        for i in range(self._n_rows):
            for j in range(self._n_cols):
                curr_loc = (i + 1, j + 1)
                if curr_loc not in self._walls:
                    locs += [' '.join(str(x) for x in curr_loc)]
        
        for loc_comb in itertools.product(locs, repeat=self._n_robots):
            add = True
            for i in range(self._n_robots):
                for j in range(i+1, self._n_robots):
                    if loc_comb[i] == loc_comb[j]:
                        add = False
            if add:
                # objs_comb = itertools.product(objs, repeat=n_robots)
                # for obj_comb in objs_comb:
                #     states += [loc_comb + (obj_comb, )]
                states += [', '.join(loc_comb)]

        return np.array(states, dtype=tuple)

    def generate_actions(self) -> np.ndarray:
        possible_actions = ['U', 'D', 'L', 'R', 'N/P']
        actions = []
        for acts_comb in itertools.product(possible_actions, repeat=self._n_robots):
            actions += [' '.join(acts_comb)]
        return np.array(actions, dtype=tuple)

    def generate_stochastic_probabilities(self, states: np.ndarray, actions: np.ndarray) -> Dict[str, np.ndarray]:

        nX = len(states)
        state_lst = list(states)
        # print(state_lst)

        # objs_loc = []
        # objs_list = []
        # for x, y, o in objs:
        #     objs_loc += [(x, y)]
        #     objs_list += [o]

        P = {}

        for act in actions:
            act_tuple = MultipleRobotMazeWorld.get_action_tuple(act)
            p = np.zeros((nX, nX))
            for state in states:
                state_tuple = MultipleRobotMazeWorld.get_state_tuple(state)
                state_idx = state_lst.index(state)
                nxt_state_tmp = ()
                # state_obj = state[-1]
                # nxt_state_obj = ()
                for i in range(self._n_robots):
                    curr_rob_loc = state_tuple[i]
                    # curr_rob_obj = state_obj[i]
                    if act_tuple[i] == 'U':
                        # nxt_state_obj += (curr_rob_obj,)
                        
                        nxt_loc = (min(self._n_rows, curr_rob_loc[0] + 1), curr_rob_loc[1])
                        if self.wall_exists(nxt_loc, act_tuple[i]):
                            nxt_state_tmp += (curr_rob_loc, )
                        else:
                            nxt_state_tmp += (nxt_loc, )
    
                    elif act_tuple[i] == 'D':
                        # nxt_state_obj += (curr_rob_obj,)
                        
                        nxt_loc = (max(1, curr_rob_loc[0] - 1), curr_rob_loc[1])
                        if self.wall_exists(nxt_loc, act_tuple[i]):
                            nxt_state_tmp += (curr_rob_loc,)
                        else:
                            nxt_state_tmp += (nxt_loc,)
        
                    elif act_tuple[i] == 'L':
                        # nxt_state_obj += (curr_rob_obj,)

                        nxt_loc = (curr_rob_loc[0], max(1, curr_rob_loc[1] - 1))
                        if self.wall_exists(nxt_loc, act_tuple[i]):
                            nxt_state_tmp += (curr_rob_loc,)
                        else:
                            nxt_state_tmp += (nxt_loc,)
        
                    elif act_tuple[i] == 'R':
                        # nxt_state_obj += (curr_rob_obj,)

                        nxt_loc = (curr_rob_loc[0], min(self._n_cols, curr_rob_loc[1] + 1))
                        if self.wall_exists(nxt_loc, act_tuple[i]):
                            nxt_state_tmp += (curr_rob_loc,)
                        else:
                            nxt_state_tmp += (nxt_loc,)
        
                    elif act_tuple[i] == 'N/P':
                        # if curr_rob_loc in objs_loc:
                        #     nxt_state_obj += (objs_list[objs_loc.index(curr_rob_loc)])
                        
                        nxt_state_tmp += (state_tuple[i], )
        
                    else:
                        print(colored('Action not recognized. Skipping matrix probability', 'red'))
                        continue
                
                # nxt_state += (nxt_state_obj, )
                can_act = np.ones(self._n_robots)
                for i in range(self._n_robots):
                    for j in range(i + 1, self._n_robots):
                        if nxt_state_tmp[i] == nxt_state_tmp[j]:
                            can_act[i] = 0
                            can_act[j] = 0
                nxt_state = ()
                for i in range(self._n_robots):
                    if can_act[i] > 0:
                        nxt_state += (nxt_state_tmp[i], )
                    else:
                        nxt_state += (state_tuple[i], )
                nxt_idx = state_lst.index(MultipleRobotMazeWorld.get_state_str(nxt_state))
                p[state_idx][nxt_idx] += (1.0 - self._fail_chance)
                p[state_idx][state_idx] += self._fail_chance

            P[act] = p

        return P

    def generate_condensed_probabilities(self, states: np.ndarray, actions: np.ndarray) -> Dict[str, np.ndarray]:
        
        state_lst = list(states)
        P = {}
        
        for act in actions:
            
            act_tuple = MultipleRobotMazeWorld.get_action_tuple(act)
            p = []
            for state in states:
                state_tuple = MultipleRobotMazeWorld.get_state_tuple(state)
                state_idx = state_lst.index(state)
                nxt_state_tmp = ()
                # state_obj = state[-1]
                # nxt_state_obj = ()
                for i in range(self._n_robots):
                    curr_rob_loc = state_tuple[i]
                    # curr_rob_obj = state_obj[i]
                    if act_tuple[i] == 'U':
                        # nxt_state_obj += (curr_rob_obj,)
            
                        nxt_loc = (min(self._n_rows, curr_rob_loc[0] + 1), curr_rob_loc[1])
                        if self.wall_exists(curr_rob_loc, act_tuple[i]):
                            nxt_state_tmp += (curr_rob_loc,)
                        else:
                            nxt_state_tmp += (nxt_loc,)
        
                    elif act_tuple[i] == 'D':
                        # nxt_state_obj += (curr_rob_obj,)
            
                        nxt_loc = (max(1, curr_rob_loc[0] - 1), curr_rob_loc[1])
                        if self.wall_exists(curr_rob_loc, act_tuple[i]):
                            nxt_state_tmp += (curr_rob_loc,)
                        else:
                            nxt_state_tmp += (nxt_loc,)
        
                    elif act_tuple[i] == 'L':
                        # nxt_state_obj += (curr_rob_obj,)
            
                        nxt_loc = (curr_rob_loc[0], max(1, curr_rob_loc[1] - 1))
                        if self.wall_exists(curr_rob_loc, act_tuple[i]):
                            nxt_state_tmp += (curr_rob_loc,)
                        else:
                            nxt_state_tmp += (nxt_loc,)
        
                    elif act_tuple[i] == 'R':
                        # nxt_state_obj += (curr_rob_obj,)
            
                        nxt_loc = (curr_rob_loc[0], min(self._n_cols, curr_rob_loc[1] + 1))
                        if self.wall_exists(curr_rob_loc, act_tuple[i]):
                            nxt_state_tmp += (curr_rob_loc,)
                        else:
                            nxt_state_tmp += (nxt_loc,)
        
                    elif act_tuple[i] == 'N/P':
                        # if curr_rob_loc in objs_loc:
                        #     nxt_state_obj += (objs_list[objs_loc.index(curr_rob_loc)])
            
                        nxt_state_tmp += (state_tuple[i],)
        
                    else:
                        print(colored('Action not recognized. Skipping matrix probability', 'red'))
                        continue
    
                # nxt_state += (nxt_state_obj, )
                can_act = np.ones(self._n_robots)
                for i in range(self._n_robots):
                    for j in range(i + 1, self._n_robots):
                        if nxt_state_tmp[i] == nxt_state_tmp[j]:
                            can_act[i] = 0
                            can_act[j] = 0
                nxt_state = ()
                for i in range(self._n_robots):
                    if can_act[i] > 0:
                        nxt_state += (nxt_state_tmp[i],)
                    else:
                        nxt_state += (state_tuple[i],)
                nxt_idx = state_lst.index(MultipleRobotMazeWorld.get_state_str(nxt_state))
                p.append([(nxt_idx, 1 - self._fail_chance), (state_idx, self._fail_chance)])
        
            P[act] = np.array(p)
            
        return P

    def get_transition(self, state_str: str, action_str: str) -> np.ndarray:
    
        state_tuple = MultipleRobotMazeWorld.get_state_tuple(state_str)
        act_tuple = MultipleRobotMazeWorld.get_action_tuple(action_str)
        nX = len(self._states)
        state_lst = list(self._states)
        p = np.zeros(nX)
        nxt_state_tmp = ()
        state_idx = state_lst.index(state_str)

        for i in range(self._n_robots):
            curr_rob_loc = state_tuple[i]
            # curr_rob_obj = state_obj[i]
            if act_tuple[i] == 'U':
                # nxt_state_obj += (curr_rob_obj,)
        
                nxt_loc = (min(self._n_rows, curr_rob_loc[0] + 1), curr_rob_loc[1])
                if self.wall_exists(nxt_loc, act_tuple[i]):
                    nxt_state_tmp += (curr_rob_loc,)
                else:
                    nxt_state_tmp += (nxt_loc,)
    
            elif act_tuple[i] == 'D':
                # nxt_state_obj += (curr_rob_obj,)
        
                nxt_loc = (max(1, curr_rob_loc[0] - 1), curr_rob_loc[1])
                if self.wall_exists(nxt_loc, act_tuple[i]):
                    nxt_state_tmp += (curr_rob_loc,)
                else:
                    nxt_state_tmp += (nxt_loc,)
    
            elif act_tuple[i] == 'L':
                # nxt_state_obj += (curr_rob_obj,)
        
                nxt_loc = (curr_rob_loc[0], max(1, curr_rob_loc[1] - 1))
                if self.wall_exists(nxt_loc, act_tuple[i]):
                    nxt_state_tmp += (curr_rob_loc,)
                else:
                    nxt_state_tmp += (nxt_loc,)
    
            elif act_tuple[i] == 'R':
                # nxt_state_obj += (curr_rob_obj,)
        
                nxt_loc = (curr_rob_loc[0], min(self._n_cols, curr_rob_loc[1] + 1))
                if self.wall_exists(nxt_loc, act_tuple[i]):
                    nxt_state_tmp += (curr_rob_loc,)
                else:
                    nxt_state_tmp += (nxt_loc,)
    
            elif act_tuple[i] == 'N/P':
                # if curr_rob_loc in objs_loc:
                #     nxt_state_obj += (objs_list[objs_loc.index(curr_rob_loc)])
        
                nxt_state_tmp += (state_tuple[i],)
    
            else:
                print(colored('Action not recognized. Skipping matrix probability', 'red'))
                continue

        can_act = np.ones(self._n_robots)
        for i in range(self._n_robots):
            for j in range(i + 1, self._n_robots):
                if nxt_state_tmp[i] == nxt_state_tmp[j]:
                    can_act[i] = 0
                    can_act[j] = 0
        nxt_state = ()
        for i in range(self._n_robots):
            if can_act[i] > 0:
                nxt_state += (nxt_state_tmp[i],)
            else:
                nxt_state += (state_tuple[i],)
        nxt_idx = state_lst.index(MultipleRobotMazeWorld.get_state_str(nxt_state))
        p[nxt_idx] += (1.0 - self._fail_chance)
        p[state_idx] += self._fail_chance
        
        return p
    
    def generate_rewards(self, goal_states: List[int]) -> np.ndarray:

        nX = len(self._states)
        nA = len(self._actions)

        c = np.zeros((nX, nA))
        for state in goal_states:
            c[state, :] = 1.0

        return c

    def generate_costs(self, goal_states: List[int]) -> np.ndarray:

        nX = len(self._states)
        nA = len(self._actions)

        c = np.ones((nX, nA))
        for state in goal_states:
            c[state, :] = 0.0

        return c

    def generate_world(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:

        print('### Generating Maze World for Multiple Robots ###')
        print('Generating States')
        states = self.generate_states()
        print('Generating Actions')
        actions = self.generate_actions()
        print('Generating Transitions')
        # probabilities = self.generate_stochastic_probabilities(states, actions)
        probabilities = self.generate_condensed_probabilities(states, actions)

        self._states = states
        self._actions = actions

        print('World Created')
        return states, actions, probabilities


class TrackedMultipleRobotMazeWorld(MultipleRobotMazeWorld):
    
    def generate_states(self) -> np.ndarray:
        states = []
        locs = []
        objs = ['N']
        for obj_state in self._obj_states:
            objs += [obj_state[2]]
        for i in range(self._n_rows):
            for j in range(self._n_cols):
                curr_loc = (i + 1, j + 1)
                if curr_loc not in self._walls:
                    locs += [' '.join(str(x) for x in curr_loc)]
        
        for loc_comb in itertools.product(locs, repeat=self._n_robots):
            add = True
            for i in range(self._n_robots):
                for j in range(i + 1, self._n_robots):
                    if loc_comb[i] == loc_comb[j]:
                        add = False
            if add:
                objs_comb = itertools.product(objs, repeat=self._n_robots)
                for comb in objs_comb:
                    states += [', '.join(loc_comb + (' '.join(comb), ))]
        
        return np.array(states, dtype=tuple)

    def generate_stochastic_probabilities(self, states: np.ndarray, actions: np.ndarray) -> Dict[str, np.ndarray]:
        nX = len(states)
        state_lst = list(states)
        # print(state_lst)
    
        # objs_loc = []
        # objs_list = []
        # for x, y, o in objs:
        #     objs_loc += [(x, y)]
        #     objs_list += [o]
    
        P = {}
    
        for act in actions:
            act_tuple = MultipleRobotMazeWorld.get_action_tuple(act)
            p = np.zeros((nX, nX))
            for state in states:
                state_tuple = MultipleRobotMazeWorld.get_state_tuple(state)
                state_idx = state_lst.index(state)
                nxt_state_tmp = ()
                # state_obj = state[-1]
                # nxt_state_obj = ()
                for i in range(self._n_robots):
                    curr_rob_loc = state_tuple[i]
                    # curr_rob_obj = state_obj[i]
                    if act_tuple[i] == 'U':
                        # nxt_state_obj += (curr_rob_obj,)
                    
                        nxt_loc = (min(self._n_rows, curr_rob_loc[0] + 1), curr_rob_loc[1])
                        if self.wall_exists(nxt_loc, act_tuple[i]):
                            nxt_state_tmp += (curr_rob_loc,)
                        else:
                            nxt_state_tmp += (nxt_loc,)
                
                    elif act_tuple[i] == 'D':
                        # nxt_state_obj += (curr_rob_obj,)
                    
                        nxt_loc = (max(1, curr_rob_loc[0] - 1), curr_rob_loc[1])
                        if self.wall_exists(nxt_loc, act_tuple[i]):
                            nxt_state_tmp += (curr_rob_loc,)
                        else:
                            nxt_state_tmp += (nxt_loc,)
                
                    elif act_tuple[i] == 'L':
                        # nxt_state_obj += (curr_rob_obj,)
                    
                        nxt_loc = (curr_rob_loc[0], max(1, curr_rob_loc[1] - 1))
                        if self.wall_exists(nxt_loc, act_tuple[i]):
                            nxt_state_tmp += (curr_rob_loc,)
                        else:
                            nxt_state_tmp += (nxt_loc,)
                
                    elif act_tuple[i] == 'R':
                        # nxt_state_obj += (curr_rob_obj,)
                    
                        nxt_loc = (curr_rob_loc[0], min(self._n_cols, curr_rob_loc[1] + 1))
                        if self.wall_exists(nxt_loc, act_tuple[i]):
                            nxt_state_tmp += (curr_rob_loc,)
                        else:
                            nxt_state_tmp += (nxt_loc,)
                
                    elif act_tuple[i] == 'N/P':
                        # if curr_rob_loc in objs_loc:
                        #     nxt_state_obj += (objs_list[objs_loc.index(curr_rob_loc)])
                    
                        nxt_state_tmp += (state_tuple[i],)
                
                    else:
                        print(colored('Action not recognized. Skipping matrix probability', 'red'))
                        continue
            
                # nxt_state += (nxt_state_obj, )
                can_act = np.ones(self._n_robots)
                for i in range(self._n_robots):
                    for j in range(i + 1, self._n_robots):
                        if nxt_state_tmp[i] == nxt_state_tmp[j]:
                            can_act[i] = 0
                            can_act[j] = 0
                nxt_state = ()
                for i in range(self._n_robots):
                    if can_act[i] > 0:
                        nxt_state += (nxt_state_tmp[i],)
                    else:
                        nxt_state += (state_tuple[i],)
                nxt_idx = state_lst.index(MultipleRobotMazeWorld.get_state_str(nxt_state))
                p[state_idx][nxt_idx] += (1.0 - self._fail_chance)
                p[state_idx][state_idx] += self._fail_chance
        
            P[act] = p
    
        return P

    def generate_condensed_probabilities(self, states: np.ndarray, actions: np.ndarray) -> Dict[str, np.ndarray]:
        
        state_lst = list(states)
        P = {}
        objs_lst = []
        objs_loc = []
        for obj in self._obj_states:
            objs_loc += [(obj[0], obj[1])]
            objs_lst += [obj[2]]
    
        for act in actions:
            act_tuple = MultipleRobotMazeWorld.get_action_tuple(act)
            p = []
            for state in states:
                state_tuple = MultipleRobotMazeWorld.get_state_tuple(state)
                state_idx = state_lst.index(state)
                nxt_state_tmp = ()
                state_obj = state_tuple[-1]
                nxt_state_obj_tmp = ()
                for i in range(self._n_robots):
                    curr_rob_loc = state_tuple[i]
                    curr_rob_obj = state_obj[i]
                    if act_tuple[i] == 'U':
                        nxt_state_obj_tmp += (curr_rob_obj,)
                    
                        nxt_loc = (min(self._n_rows, curr_rob_loc[0] + 1), curr_rob_loc[1])
                        if self.wall_exists(curr_rob_loc, act_tuple[i]):
                            nxt_state_tmp += (curr_rob_loc,)
                        else:
                            nxt_state_tmp += (nxt_loc,)
                
                    elif act_tuple[i] == 'D':
                        nxt_state_obj_tmp += (curr_rob_obj,)
                    
                        nxt_loc = (max(1, curr_rob_loc[0] - 1), curr_rob_loc[1])
                        if self.wall_exists(curr_rob_loc, act_tuple[i]):
                            nxt_state_tmp += (curr_rob_loc,)
                        else:
                            nxt_state_tmp += (nxt_loc,)
                
                    elif act_tuple[i] == 'L':
                        nxt_state_obj_tmp += (curr_rob_obj,)
                    
                        nxt_loc = (curr_rob_loc[0], max(1, curr_rob_loc[1] - 1))
                        if self.wall_exists(curr_rob_loc, act_tuple[i]):
                            nxt_state_tmp += (curr_rob_loc,)
                        else:
                            nxt_state_tmp += (nxt_loc,)
                
                    elif act_tuple[i] == 'R':
                        nxt_state_obj_tmp += (curr_rob_obj,)
                    
                        nxt_loc = (curr_rob_loc[0], min(self._n_cols, curr_rob_loc[1] + 1))
                        if self.wall_exists(curr_rob_loc, act_tuple[i]):
                            nxt_state_tmp += (curr_rob_loc,)
                        else:
                            nxt_state_tmp += (nxt_loc,)
                
                    elif act_tuple[i] == 'N/P':
                        if curr_rob_loc in objs_loc:
                            nxt_state_obj_tmp += (objs_lst[objs_loc.index(curr_rob_loc)], )
                        else:
                            nxt_state_obj_tmp += (curr_rob_obj, )

                        nxt_state_tmp += (state_tuple[i],)
                
                    else:
                        print(colored('Action not recognized. Skipping matrix probability', 'red'))
                        continue
            
                can_act = np.ones(self._n_robots)
                for i in range(self._n_robots):
                    for j in range(i + 1, self._n_robots):
                        if nxt_state_tmp[i] == nxt_state_tmp[j]:
                            can_act[i] = 0
                            can_act[j] = 0
                nxt_state = ()
                nxt_state_obj = ()
                for i in range(self._n_robots):
                    if can_act[i] > 0:
                        nxt_state += (nxt_state_tmp[i],)
                        nxt_state_obj += (nxt_state_obj_tmp[i],)
                    else:
                        nxt_state += (state_tuple[i],)
                        nxt_state_obj += (state_obj[i],)
                nxt_state += (nxt_state_obj, )
                nxt_idx = state_lst.index(MultipleRobotMazeWorld.get_state_str(nxt_state))
                p.append([(nxt_idx, 1 - self._fail_chance), (state_idx, self._fail_chance)])
        
            P[act] = np.array(p)
    
        return P

    def get_transition(self, state_str: str, action_str: str) -> np.ndarray:
        state_tuple = MultipleRobotMazeWorld.get_state_tuple(state_str)
        act_tuple = MultipleRobotMazeWorld.get_action_tuple(action_str)
        nX = len(self._states)
        state_lst = list(self._states)
        p = np.zeros(nX)
        nxt_state_tmp = ()
        state_idx = state_lst.index(state_str)
    
        for i in range(self._n_robots):
            curr_rob_loc = state_tuple[i]
            # curr_rob_obj = state_obj[i]
            if act_tuple[i] == 'U':
                # nxt_state_obj += (curr_rob_obj,)
            
                nxt_loc = (min(self._n_rows, curr_rob_loc[0] + 1), curr_rob_loc[1])
                if self.wall_exists(nxt_loc, act_tuple[i]):
                    nxt_state_tmp += (curr_rob_loc,)
                else:
                    nxt_state_tmp += (nxt_loc,)
        
            elif act_tuple[i] == 'D':
                # nxt_state_obj += (curr_rob_obj,)
            
                nxt_loc = (max(1, curr_rob_loc[0] - 1), curr_rob_loc[1])
                if self.wall_exists(nxt_loc, act_tuple[i]):
                    nxt_state_tmp += (curr_rob_loc,)
                else:
                    nxt_state_tmp += (nxt_loc,)
        
            elif act_tuple[i] == 'L':
                # nxt_state_obj += (curr_rob_obj,)
            
                nxt_loc = (curr_rob_loc[0], max(1, curr_rob_loc[1] - 1))
                if self.wall_exists(nxt_loc, act_tuple[i]):
                    nxt_state_tmp += (curr_rob_loc,)
                else:
                    nxt_state_tmp += (nxt_loc,)
        
            elif act_tuple[i] == 'R':
                # nxt_state_obj += (curr_rob_obj,)
            
                nxt_loc = (curr_rob_loc[0], min(self._n_cols, curr_rob_loc[1] + 1))
                if self.wall_exists(nxt_loc, act_tuple[i]):
                    nxt_state_tmp += (curr_rob_loc,)
                else:
                    nxt_state_tmp += (nxt_loc,)
        
            elif act_tuple[i] == 'N/P':
                # if curr_rob_loc in objs_loc:
                #     nxt_state_obj += (objs_list[objs_loc.index(curr_rob_loc)])
            
                nxt_state_tmp += (state_tuple[i],)
        
            else:
                print(colored('Action not recognized. Skipping matrix probability', 'red'))
                continue
    
        can_act = np.ones(self._n_robots)
        for i in range(self._n_robots):
            for j in range(i + 1, self._n_robots):
                if nxt_state_tmp[i] == nxt_state_tmp[j]:
                    can_act[i] = 0
                    can_act[j] = 0
        nxt_state = ()
        for i in range(self._n_robots):
            if can_act[i] > 0:
                nxt_state += (nxt_state_tmp[i],)
            else:
                nxt_state += (state_tuple[i],)
        nxt_idx = state_lst.index(MultipleRobotMazeWorld.get_state_str(nxt_state))
        p[nxt_idx] += (1.0 - self._fail_chance)
        p[state_idx] += self._fail_chance
    
        return p

