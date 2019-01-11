import argparse
import sys
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from unityagents import UnityEnvironment

from agent_ddpg import Agent


def run_ddpg(environment, agents, weights_actors, weights_critics, n_episodes=2000):
    """Run Deep Deterministic Policy Gradient Learning for the given agents in the given environment.
    
    Params
    ======
        environment (UnityEnvironment): environment
        agents (list of Agent): agents
        weights_actors (list of str): files to store the weights of the actors
        weights_critics (list of str): files to store the weights of the critics
        n_episodes (int): maximum number of training episodes
    """

    brain_name = environment.brain_names[0]
    environment_info = environment.reset(train_mode=True)[brain_name]

    agents_size = len(agents)  # number of agents
    states_size = agents_size * environment_info.vector_observations.shape[1]  # size of the state space shared by the agents

    scores = []  # scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores

    for i_episode in range(1, n_episodes + 1):
        environment_info = environment.reset(train_mode=True)[brain_name]
        states = environment_info.vector_observations.reshape((1, states_size))

        # Reset the agents
        for agent in agents:
            agent.reset()

        scores_agents = np.zeros(agents_size)

        while True:

            # Perform actions in the environment
            actions = [agent.act(states, True) for agent in agents]  # execute actions with added noise
            actions = np.hstack(tuple(actions))  # stack the actions performed by the agents

            environment_info = environment.step(actions)[brain_name]  # send both agents' actions together to the environment
            next_states = environment_info.vector_observations.reshape((1, states_size))

            rewards = environment_info.rewards  # get reward
            dones = environment_info.local_done  # verify if episode finished

            for i, agent in enumerate(agents):
                agent.step(states, actions, rewards[i], next_states, dones[i], i)  # agent i learns

            scores_agents += rewards  # update the score for each agent
            states = next_states  # roll over states to next time step

            if np.any(dones):
                break

        scores_window.append(np.max(scores_agents))
        scores.append(np.max(scores_agents))

        if i_episode % 10 == 0:
            print('Episode {}\tMax reward: {:.3f}\tAverage reward: {:.3f}'.format(i_episode, np.max(scores_agents), np.mean(scores_window)))

        if np.mean(scores_window) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage score: {:.3f}'.format(i_episode - 100, np.mean(scores_window)))

            for agent, weights_actor, weights_critic in zip(agents, weights_actors, weights_critics):
                torch.save(agent.actor_local.state_dict(), weights_actor)
                torch.save(agent.critic_local.state_dict(), weights_critic)

            break

    for agent, weights_actor, weights_critic in zip(agents, weights_actors, weights_critics):
        torch.save(agent.actor_local.state_dict(), weights_actor)
        torch.save(agent.critic_local.state_dict(), weights_critic)

    return scores


def plot_scores(scores, plot_name):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')

    window_mean = pd.Series(scores).rolling(100).mean()
    plt.plot(window_mean, linewidth=4)

    plt.savefig(plot_name)
    plt.show()


def setup_environment(file_name):
    environment = UnityEnvironment(file_name=file_name)
    brain_name = environment.brain_names[0]
    brain = environment.brains[brain_name]
    environment_info = environment.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state_size = environment_info.vector_observations.shape[1]

    return environment, action_size, state_size


def main(arguments):
    parameters = parse_arguments(arguments)

    # Specify the path to the environment
    file_name = 'Tennis_Linux/Tennis.x86_64'

    # Set up the environment
    environment, action_size, state_size = setup_environment(file_name)

    # Set up the agents
    agents = [
        Agent(state_size=state_size, action_size=action_size, random_seed=0),
        Agent(state_size=state_size, action_size=action_size, random_seed=0)
    ]

    # Retrieve weights for the actors
    weights_actors = [
        parameters.actor1,
        parameters.actor2
    ]

    # Retrieve weights for the critics
    weights_critics = [
        parameters.critic1,
        parameters.critic2
    ]

    # Run the Deep Deterministic Policy Gradient Learning algorithm
    scores = run_ddpg(environment=environment, agents=agents, weights_actors=weights_actors, weights_critics=weights_critics, n_episodes=parameters.episodes)

    # Plot the scores
    plot_scores(scores=scores, plot_name=parameters.plot)

    # Close the environment
    environment.close()


def parse_arguments(arguments):
    parser = argparse.ArgumentParser(description='Run the Tennis trainer.')
    parser.add_argument('--episodes', '-e', required=False, type=int, default=2000, help='Number of episodes to run.')
    parser.add_argument('--actor1', required=True, type=str, help='Path to a file to store the network weights for the first actor.')
    parser.add_argument('--actor2', required=True, type=str, help='Path to a file to store the network weights for the second actor.')
    parser.add_argument('--critic1', required=True, type=str, help='Path to a file to store the network weights for the first critic.')
    parser.add_argument('--critic2', required=True, type=str, help='Path to a file to store the network weights for the second critic.')
    parser.add_argument('--plot', '-p', required=False, type=str, default='plot.png', help='Path to a file to store a plot of the scores.')

    return parser.parse_args(arguments)


if __name__ == '__main__':
    main(sys.argv[1:])
