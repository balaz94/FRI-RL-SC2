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
    name = 'zergs'
    optim = 'Adam'
    device = "cuda"
    results_path = 'results/zergs/'


    agent = Agent(PolicyValueModel(32, 2048), gamma, entropy_loss_coef, value_loss_coef ,epsilon, lr, name, optim, device,results_path)

    count_of_processes = 3
    count_of_envs = 1
    count_of_steps = 4096
    batch_size = 4096

    count_of_epochs = 4
    input_dim = (3, 64, 64)


    agent.load_model('results/zergs/models/ppo41_ppo.pt')
    agent.train("", Env, count_of_processes, count_of_envs, count_of_iterations, count_of_steps, batch_size, count_of_epochs, first_iteration, input_dim)





   
def animation():
    model = PolicyValueModel(32, 2048) # zergs
    model.to("cuda")
    model.load_state_dict(torch.load("results/zergs/models/best.pt"))

    env = Env(*"")
    i = 0
    while i < 10:
        terminal = False
        i += 1
        observations = torch.from_numpy(env.reset())
        while not terminal:
            with torch.no_grad():
                observations = observations.to("cuda").unsqueeze(0).float()
                logits, values = model(observations)

                #logits, _, _ = model.policy_model(state)
                probs = F.softmax(logits, dim=-1)
                action = probs.multinomial(num_samples=1)

            new_state,_,terminal,_ = env.step(action[0,0].item())
            observations = torch.from_numpy(new_state)


if __name__ == "__main__":
   learning(200)
    # graph.make_graph.scatter_plot('results/zergs/data/ppo.csv')
    # animation()




