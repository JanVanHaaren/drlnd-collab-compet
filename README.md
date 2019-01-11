# Project 3: Collaboration and Competition

## Introduction

The goal of this project is to train two agents that control rackets to bounce a ball over a net. An agent receives a reward of +0.1 if it hits the ball over the net. An agent receives a reward of -0.01 if it lets the ball hit the ground or hits the ball out of bounds. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own local observation. Two continuous actions are available, corresponding to movement toward or away from the net, and jumping. 

The agents must get an average score of +0.5 over 100 consecutive episodes to solve the environment.


## Setting up the project

0. Clone this repository.

1. Download the environment from one of the links below. Select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip);
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip);
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip);
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip).
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.

2. Place the file in the repository folder and unzip (or decompress) the file. 

3. Create a Conda environment containing the packages listed in the `environment.yml` file as follows:
    ```bash
    $ conda env install -f environment.yml
    ```

4. Activate the created `drlnd-collab-compet` environment as follows:

    ```bash
    # Linux and Mac OSX
    $ source activate drlnd-collab-compet
    
    # Windows
    $ activate drlnd-collab-compet
    ```

## Training and testing the agents

### Training the agents

Run the `agent_train.py` script as follows:

```bash
$ python agent_train.py --episodes 2000 --actor1 weights_actor1.pth --actor2 weights_actor2.pth --critic1 weights_critic1.pth--critic2 weights_critic2.pth --plot plot.png
```

The script accepts the following parameters:
- **`--episodes`** (default: 2000): the maximum number of episodes;
- **`--actor1`**: name of the file to store the weights of the first actor network;
- **`--actor2`**: name of the file to store the weights of the second actor network;
- **`--critic1`**: name of the file to store the weights of the first critic network;
- **`--critic2`**: name of the file to store the weights of the second critic network;
- **`--plot`** (default: plot.png): name of the file to store the plot of the obtained scores.

### Testing the agents

Run the `agent_test.py` script as follows:

```bash
$ python agent_test.py --actor1 weights_actor1.pth --actor2 weights_actor2.pth --critic1 weights_critic1.pth--critic2 weights_critic2.pth
```

The script accepts the following parameters:
- **`--actor1`**: name of the file to load the weights of the first actor network;
- **`--actor2`**: name of the file to load the weights of the second actor network;
- **`--critic1`**: name of the file to load the weights of the first critic network;
- **`--critic2`**: name of the file to load the weights of the second critic network.
