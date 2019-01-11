import argparse
import sys

import numpy as np
import torch
from unityagents import UnityEnvironment

from agent_ddpg import Agent


def run_agents(environment, agents):
    brain_name = environment.brain_names[0]
    environment_info = environment.reset(train_mode=False)[brain_name]

    agents_size = len(agents)  # number of agents
    states_size = agents_size * environment_info.vector_observations.shape[1]  # size of the state space shared by the agents

    states = environment_info.vector_observations.reshape((1, states_size))
    scores = np.zeros(agents_size)

    while True:

        actions = [agent.act(states, False) for agent in agents]  # execute actions
        actions = np.hstack(tuple(actions))  # stack the actions performed by the agents

        environment_info = environment.step(actions)[brain_name]  # send both agents' actions together to the environment
        next_states = environment_info.vector_observations.reshape((1, states_size))

        rewards = environment_info.rewards
        dones = environment_info.local_done

        scores += rewards
        states = next_states

        if np.any(dones):
            break

    print('Score: {}'.format(np.mean(scores)))


def setup_environment(file_name):
    environment = UnityEnvironment(file_name=file_name)
    brain_name = environment.brain_names[0]
    brain = environment.brains[brain_name]
    environment_info = environment.reset(train_mode=False)[brain_name]
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
        Agent(state_size=state_size, action_size=action_size, random_seed=0),
    ]

    # Load the network weights
    agents[0].actor_local.load_state_dict(torch.load('{}'.format(parameters.actor1)))
    agents[1].actor_local.load_state_dict(torch.load('{}'.format(parameters.actor2)))

    agents[0].critic_local.load_state_dict(torch.load('{}'.format(parameters.critic1)))
    agents[1].critic_local.load_state_dict(torch.load('{}'.format(parameters.critic2)))

    # Run the agents
    run_agents(environment=environment, agents=agents)

    # Close the environment
    environment.close()


def parse_arguments(arguments):
    parser = argparse.ArgumentParser(description='Run the Tennis tester.')
    parser.add_argument('--actor1', required=True, type=str, help='Path to a file to load the network weights for the first actor.')
    parser.add_argument('--actor2', required=True, type=str, help='Path to a file to load the network weights for the second actor.')
    parser.add_argument('--critic1', required=True, type=str, help='Path to a file to load the network weights for the first critic.')
    parser.add_argument('--critic2', required=True, type=str, help='Path to a file to load the network weights for the second critic.')

    return parser.parse_args(arguments)


if __name__ == '__main__':
    main(sys.argv[1:])
