from collections import deque

import plotly.graph_objects as go
import pandas as pd

from pyglet.resource import animation

from envs.sc2_find_and_defeat_zerglings_env import Env
from models.sc2_zergs import PolicyValueModel
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import graph.make_graph
from agents.ppo_parallel import Agent



MAX_SCORE_COUNT = 100

def learning(count_of_iterations):
    gamma = 0.99
    entropy_loss_coef = 0.001
    value_loss_coef = 0.5
    epsilon = 0.1
    
    lr = 0.00025
    name = 'ppo'
    optim = 'Adam'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_path = 'results/zergs/'

    agent = Agent(PolicyValueModel(36, 2048), gamma, entropy_loss_coef, value_loss_coef ,epsilon, lr, name, optim, device,results_path)

    count_of_processes = 3
    count_of_envs = 2
    count_of_steps = 1500
    batch_size = 500
    count_of_epochs = 4
    first_iteration = 0
    input_dim = (3, 64, 64)

    agent.train("", Env, count_of_processes, count_of_envs, count_of_iterations, count_of_steps, batch_size, count_of_epochs, first_iteration, input_dim)


if __name__ == "__main__":
   learning(2000)
   # graph.make_graph.scatter_plot('results/zergs/data/ppo.csv')



