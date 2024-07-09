import random
from collections import deque
import numpy as np
import torch
import gymnasium as gym

from agents.agent import Agent

def train():
    env = gym.make('MsPacmanDeterministic-v0', full_action_space = False)
    state_shape = env.observation_space.shape
    state_size = env.observation_space.shape[0]
    number_actions = env.action_space.n

    print('State shape: ', state_shape)
    print('State size: ', state_size)
    print('Number of actions: ', number_actions)

    agent = Agent(number_actions)

    number_episodes = 2000
    maximum_number_timesteps_per_episode = 1000
    epsilon_starting_value = 1.0
    epsilon_ending_value = 0.01
    epsilon_decay_value = 0.995
    epsilon = epsilon_starting_value
    scores_on_100_episodes = deque(maxlen=100)

    for episode in range(1, number_episodes + 1):
        state, _ = env.reset()
        score = 0

        for t in range(maximum_number_timesteps_per_episode):
            action = agent.act(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break
        scores_on_100_episodes.append(score)
        epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)), end="")

        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))
            
        if np.mean(scores_on_100_episodes) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100, np.mean(scores_on_100_episodes)))
            torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
            break
    
    return agent