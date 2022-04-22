#! /usr/bin/env python

import pytest
import numpy as np
import gym
import lbforaging
import time
import argparse

from lbforaging.foraging.environment import Action
from agents.lb_foraging_agent import LBForagingQAgent, Timestep
from policies import Policies
from learners.q_learner import QLearner
from pathlib import Path
from tqdm import tqdm

LEARNING_RATE = 0.3
DISCOUNT_FACTOR = 0.9
EPS = 0.25
DECAY = 0.95
SPAWN_THRESH = 0.75
ENV_SEED = 2022


def update_fruits(env: lbforaging.foraging.environment, rng: np.random.Generator, max_fruits: int, max_food_level: int, spawn_thresh: float):
	
	fruits_spawned = env.field.sum()
	if fruits_spawned < 1:
		env.spawn_food(max_fruits, max_food_level)
	elif fruits_spawned < max_fruits:
		if rng.random() > spawn_thresh:
			env.spawn_food(max_fruits, max_food_level)
	else:
		pass


def train_agents(n_runs: int, n_iterations: int):
	env = gym.make("Foraging-8x8-2p-1f-v2")
	agent_rng_gen = np.random.default_rng(int(time.time()))
	agents = ()
	actions = [action.value for action in Action]
	for i in range(len(env.players)):
		agent_model = QLearner(actions, LEARNING_RATE, DISCOUNT_FACTOR, 'rewards')
		agent = LBForagingQAgent(Policies.eps_greedy, agent_model, 1, 2, i)
		agents += (agent, )
	
	for run in range(n_runs):
		print('RUN: %d of %d' % (run + 1, n_runs))
		env_rng_gen = np.random.default_rng(ENV_SEED)
		eps = EPS
		env.seed(ENV_SEED)
		init_state = env.reset()
		field = env.field
		action = ()
		for agent in agents:
			agent_idx = agent.agent_idx
			t_action = np.random.choice(len(actions))
			agent.state = agent.get_state_from_obs((init_state[agent_idx], field))
			agent.action = t_action
			action += (t_action, )
		max_food_level = 0
		for player in env.players:
			max_food_level += player.level
		
		for t in tqdm(range(n_iterations)):
			# env.render()
			observation, reward, done, info = env.step(action)
			field = env.field
			nxt_action = ()
			for agent in agents:
				agent_idx = agent.agent_idx
				t_observation = (observation[agent_idx], field)
				timestep = Timestep(agent.state, agent.action, t_observation, reward[agent_idx], done[agent_idx], agent_rng_gen, (eps, ))
				nxt_action += (agent.step(timestep), )
				# if sum(reward) > 0:
				# 	print('T: ' + str(t))
				# 	print(agent.learner_model.Q.to_numpy())
		
			update_fruits(env, env_rng_gen, env.max_food, max_food_level, SPAWN_THRESH)
			action = nxt_action
			# eps *= (1 - DECAY**t * eps)
			# print(eps)
	
	env.close()
	for agent in agents:
		print(sum(agent.learner_model.Q.to_numpy()))
		agent.learner_model.save('lb-foraging-agent-' + str(agents.index(agent)))


def eval_agents(n_runs: int):
	env = gym.make("Foraging-8x8-2p-1f-v2")
	init_state = env.reset()
	field = env.field
	agent_rng_gen = np.random.default_rng(int(time.time()))
	env_rng_gen = np.random.default_rng(int(time.time() / 100))
	eps = 0.0
	agents = ()
	action = ()
	actions = [action.value for action in Action]
	for i in range(len(env.players)):
		agent_model = QLearner(actions, LEARNING_RATE, DISCOUNT_FACTOR, 'rewards')
		agent_model.load('lb-foraging-agent-' + str(i))
		agent = LBForagingQAgent(Policies.eps_greedy, agent_model, 1, 2, i)
		agent.state = agent.get_state_from_obs((init_state[i], field))
		agents += (agent,)
		action += (agent.new_action(agent_model.Q.loc[agent.state].to_numpy(), agent_rng_gen, (eps, )))
	max_food_level = 0
	for player in env.players:
		max_food_level += player.level
	for _ in range(1000):
		env.render()
		observation, reward, done, info = env.step(action)
		
		if np.array(done).all():
			return

		field = env.field
		action = ()
		for agent in agents:
			agent_idx = agent.agent_idx
			agent.state = agent.get_state_from_obs((observation[agent_idx], field))
			action += (agent.new_action(agent.learner_model.Q.loc[agent.state].to_numpy(), agent_rng_gen, (eps,)))
	env.close()
	

def main():
	parser = argparse.ArgumentParser(description='LB-Foraging Team Legibility')
	parser.add_argument('--mode', dest='mode', type=str, required=True, choices=['train', 'eval'],
						help='Mode to run the application: either train to train a model or eval to check model\'s performance')
	parser.add_argument('--runs', dest='n_runs', type=int, required=True,
						help='Number of times to run the environment')
	parser.add_argument('--iterations', dest='n_iterations', type=int, required=False,
						help='Number of iterations for each training run')
	args = parser.parse_args()
	
	if args.mode.find('train') != -1:
		train_agents(args.n_runs, args.n_iterations)
	elif args.mode.find('eval') != -1:
		eval_agents(args.n_runs)
	else:
		print('Invalid execution mode')
	

if __name__ == '__main__':
	main()
