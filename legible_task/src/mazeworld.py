import numpy as np
import re
from termcolor import colored
from abc import ABC
from itertools import combinations, permutations


class MazeWorld(ABC):

    def generate_states(self, n_rows, n_cols, obj_states):
        pass

    def generate_actions(self):
        pass

    def generate_probabilities(self, states, actions, obj_states, max_rows, max_cols):
        pass

    def generate_costs(self, goal, states, actions):
        pass

    def generate_costs_varied(self, goal, states, actions, probabilities):
        pass

    def generate_world(self, n_rows, n_cols, obj_states):
        states = self.generate_states(n_rows, n_cols, obj_states)
        actions = self.generate_actions()
        probabilities = self.generate_probabilities(states, actions, obj_states, n_rows, n_cols)

        return states, actions, probabilities


class WallMazeWorld(ABC):

    def wall_exists(self, state, action, walls):
        pass

    def generate_states(self, n_rows, n_cols, obj_states):
        pass

    def generate_actions(self):
        pass

    def generate_probabilities(self, states, actions, obj_states, max_rows, max_cols, walls):
        pass

    def generate_costs(self, goal, states, actions):
        pass

    def generate_costs_varied(self, goal, states, actions, probabilities):
        pass

    def generate_world(self, n_rows, n_cols, obj_states, walls):
        states = self.generate_states(n_rows, n_cols, obj_states)
        actions = self.generate_actions()
        probabilities = self.generate_probabilities(states, actions, obj_states, n_rows, n_cols, walls)

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

    def generate_costs(self, goal, states, actions):

        nX = len(states)
        nA = len(actions)

        c = np.ones((nX, nA))

        for state in states:
            if state.find(goal) != -1:
                c[list(states).index(state), :] = 0.0

        return c

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


class WallAutoCollectMazeWorld(WallMazeWorld):

    def wall_exists(self, state, action, walls):

        state_split = re.match(r"([0-9]+) ([0-9]+) ([a-zA-z]+)", state, re.I)
        state_row = int(state_split.group(1))
        state_col = int(state_split.group(2))

        for wall in walls:
            if action == 'U':
                wall_left = (state_row - 0.5, state_col - 0.5)
                wall_right = (state_row - 0.5, state_col + 0.5)
                if wall_left in wall and wall_right in wall:
                    return True

            elif action == 'D':
                wall_left = (state_row + 0.5, state_col - 0.5)
                wall_right = (state_row + 0.5, state_col + 0.5)
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

    def generate_probabilities(self, states, actions, obj_states, max_rows, max_cols, walls):

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

    def generate_costs(self, goal, states, actions):

        nX = len(states)
        nA = len(actions)

        c = np.ones((nX, nA))

        for state in states:
            if state.find(goal) != -1:
                c[list(states).index(state), :] = 0.0

        return c

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
                    tmp_nxt_row = max(1, curr_row - 1)
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
                    tmp_nxt_row = min(max_rows, curr_row + 1)
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

    def generate_costs(self, goal, states, actions):

        nX = len(states)
        nA = len(actions)

        c = np.ones((nX, nA))

        for state in states:
            if state.find(goal) != -1:
                c[list(states).index(state), :] = 0.0

        return c

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
