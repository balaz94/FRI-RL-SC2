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

    agent = Agent(PolicyValueModel(8, 1936), gamma, entropy_loss_coef, value_loss_coef ,epsilon, lr, name, optim, device,results_path)

    count_of_processes = 3
    count_of_envs = 2
    count_of_steps = 1250
    batch_size = 750
    count_of_epochs = 4
    first_iteration = 0
    input_dim = (2, 84, 84)

    agent.load_model('results/beacon/models/ppo1_ppo.pt')
    agent.train("", Env, count_of_processes, count_of_envs, count_of_iterations, count_of_steps, batch_size, count_of_epochs, first_iteration, input_dim)


if __name__ == "__main__":
    learning(2000)
