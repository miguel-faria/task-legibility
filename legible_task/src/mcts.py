#! /usr/bin/env python

from __future__ import annotations
import numpy as np

from tqdm import tqdm
from typing import List, Dict, Tuple
from abc import ABC
from mdp import MDP, MiuraLegibleMDP


class MCTSMDPNode(ABC):
	
	def __init__(self, num_actions: int, node_state: int, mdp: MDP, parent_node=None, terminal: bool = False):
		self._num_actions = num_actions
		self._q = np.zeros(self._num_actions)
		self._n = np.ones(self._num_actions)
		self._state = node_state
		self._parent = parent_node
		self._children = dict()
		self._terminal = terminal
		self._mdp = mdp
		
	def expand(self, action: str, depth: int, discount: float, goal_states: List[int], pol: np.ndarray) -> (float, List[Tuple]):

		act_idx = self._mdp.mdp[1].index(action)
		next_node, reward = self.simulate_action(action, goal_states)
		
		if self._terminal:
			action_reward = reward
			history = [(self._state, )]
		
		else:
			future_reward, future_choices = next_node.rollout(depth - 1, discount, goal_states, pol)
			action_reward = reward + discount * future_reward
			history = [(self._state, action)] + future_choices
		
		self._n[act_idx] += 1
		self._q[act_idx] += (action_reward - self._q[act_idx]) / self._n[act_idx]
		
		return action_reward, history
	
	def rollout(self, depth: int, discount: float, goal_states: List[int], pol: np.ndarray) -> (float, List[Tuple]):
		
		if depth == 0:
			return 0.0
		
		action = np.random.choice(range(self._num_actions), p=pol[self._state])
		next_node, reward = self.simulate_action(action, goal_states)
		
		if self._terminal:
			return reward, [(self._state, )]
		else:
			future_reward, future_choices = next_node.rollout(depth - 1, discount, goal_states, pol)
			return reward + discount * future_reward, [(self._state, action)] + future_choices

	def simulate_action(self, action: str, goal_states: List[int]) -> (int, float):
		
		nX = len(self._mdp.mdp[0])
		A = self._mdp.mdp[1]
		act_idx = A.index(action)
		P = self._mdp.mdp[2]
		c = self._mdp.mdp[3]
		
		next_state = np.random.choice(nX, p=P[action][self._state, :])
		reward = c[next_state, act_idx]

		terminal_state = next_state in goal_states
		next_node = MCTSMDPNode(self._num_actions, next_state, self._mdp, self, terminal_state)
		return next_node, reward
	
	def simulate(self, depth: int, exploration: float, discount_factor: float, visited_nodes: List['MCTSMDPNode'],
				 goal_states: List[int], pol: np.ndarray) -> float:
		
		A = self._mdp.mdp[1]
		if depth == 0:
			return 0.0
		
		if self not in visited_nodes:
			visited_nodes.append(self)
			for act_idx in range(self._num_actions):
				self.expand(A[act_idx], depth, discount_factor, goal_states, pol)
			node = self
		else:
			node = visited_nodes[visited_nodes.index(self)]
		
		act_idx = node.ucb_alternative().argmax()
		q, history = self.expand(A[act_idx], depth, discount_factor, goal_states, pol)
		
		return q

	def ucb(self, c: float) -> np.ndarray:
		return self._q + c * np.sqrt(2 * np.log(self._n.sum()) / self._n)
	
	def ucb_alternative(self) -> np.ndarray:
		return self._q + self._q * np.sqrt(2 * np.log(self._n.sum()) / self._n)

	def uct(self, goal_states: List[int], max_iterations: int, max_depth: int, exploration: float, pol: np.ndarray) -> int:
		
		visited_nodes = []
		for _ in tqdm(range(max_iterations)):
			self.simulate(max_depth, exploration, self._mdp.mdp[4], visited_nodes, goal_states, pol)
			
		return self._q.argmax()
	
	
class MiuraMDPMCTSNode(MCTSMDPNode):
	
	def __init__(self, num_actions: int, node_state: int, mdp: MiuraLegibleMDP, belief: np.ndarray,
				 parent_node=None, terminal: bool = False):
		
		super(MiuraMDPMCTSNode, self).__init__(num_actions, node_state, mdp, parent_node, terminal)
		self._belief = belief
		self._objectives = mdp.tasks
		self._objective = mdp.goal
		
	def simulate_action(self, action: str, goal_states: List[int]) -> (int, float):
		nX = len(self._mdp.mdp[0])
		P = self._mdp.mdp[2]
		
		next_state = np.random.choice(nX, p=P[action][self._state, :])
		belief = self._mdp.update_belief(self._state, action, next_state, self._belief)
		reward = self._mdp.belief_reward(belief)
		
		terminal_state = next_state in goal_states
		next_node = MiuraMDPMCTSNode(self._num_actions, next_state, self._mdp, belief, self, terminal_state)
		return next_node, reward

