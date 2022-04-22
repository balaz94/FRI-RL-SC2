from collections import deque

import plotly.graph_objects as go
import pandas as pd

from pyglet.resource import animation

from envs.sc2_marines_vs_zealot import Env
from models.sc2_marines_vs_zealot import PolicyValueModel
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import graph.make_graph
from agents.ppo_parallel import Agent
import graph.make_graph as graph
import datetime
import os
import signal
import torch.nn.init as init
import time


MAX_SCORE_COUNT = 100

START_TIME = 0
total_time_file = None
score_file = None
steps_file = None
agent = None

def learning(count_of_iterations):
    gamma = 0.99
    entropy_loss_coef = 0.001
    value_loss_coef = 0.5
    epsilon = 0.1
    lr = 0.01
    name = 'ppo'
    optim = 'Adam'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_path = 'results/marines_vs_zealot/'

    actions = 27

    #agent = PolicyValueModel(0.99, actions, Net(), 0.001, beta_entropy=0.01, id=0, name='sc2_custom_ZealotArmorUp_re_mc_10_ntr', depth = 21, height = 64, width = 64)
    agent = Agent(PolicyValueModel(27, 800), gamma, entropy_loss_coef, value_loss_coef ,epsilon, lr, name, optim, device,results_path)
    agent.model.load_state_dict(torch.load('results/marines_vs_zealot/models/ppo556_ppo.pt'))

    # run_id = int(datetime.datetime.timestamp(datetime.datetime.now()))
    # os.mkdir("logs/sc2/custom_ZealotArmorUp/" + str(run_id))
    # total_time_file = open("logs/sc2/custom_ZealotArmorUp/" + str(run_id) + "/Train_time.txt", "w", buffering=1)
    # START_TIME = datetime.datetime.now()
    # signal.signal(signal.SIGTERM, at_exist)
    # signal.signal(signal.SIGINT, at_exist)

    count_of_processes = 1
    count_of_envs = 1
    count_of_steps = 1024
    batch_size = 1024
    count_of_epochs = 4
    first_iteration = 0
    input_dim = (29, 64, 64)

    graph.scatter_plot("results/marines_vs_zealot/data/ppoF1.txt")
    graph.scatter_plot("results/marines_vs_zealot/data/ppoF2.txt")
    agent.train("", Env, count_of_processes, count_of_envs, count_of_iterations, count_of_steps, batch_size, count_of_epochs, first_iteration, input_dim)
    #agent.load_model("results/marines_vs_zealot/models/ppoF1.pt")
    #agent.load_model("results/marines_vs_zealot/models/ppoF2.pt")

def animation():
    model = PolicyValueModel(27, 800)  # shards
    model.to("cuda")

    model.load_state_dict(torch.load("results/marines_vs_zealot/models/ppoF2.pt"))

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

                probs = F.softmax(logits, dim=-1)
                action = probs.multinomial(num_samples=1)

            new_state, _, terminal = env.step(action[0, 0].item())
            observations = torch.from_numpy(new_state)


if __name__ == '__main__':
    #learning(2000)
    animation()