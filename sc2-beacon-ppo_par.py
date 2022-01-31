from collections import deque

from pyglet.resource import animation

from envs.sc2_beacon_env import Env
from models.sc2_beacon import PolicyValueModel
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.ppo_parallel import Agent
from wrapper.atari_wrapper import make_env



MAX_SCORE_COUNT = 100

def learning(count_of_iterations):
    gamma = 0.99
    entropy_loss_coef = 0.001
    value_loss_coef = 0.5
    epsilon = 0.1
    
    lr = 0.00025
    name = 'beacon'
    optim = 'Adam'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_path = 'results/beacon/'

    count_of_actions = 8
    agent = Agent(PolicyValueModel(count_of_actions, 1936), optim, gamma, epsilon, 0.001, 0.5, 0.95, name, results_path, device)

    count_of_processes = 1
    count_of_envs = 1
    count_of_steps = 1250
    batch_size = 750
    count_of_epochs = 4
    first_iteration = 0

    input_dim = (2, 84, 84)
    agent.train("", Env, count_of_actions, count_of_iterations, count_of_processes, count_of_envs, count_of_steps,
                count_of_epochs, batch_size, input_dim)



if __name__ == "__main__":
    learning(200)
