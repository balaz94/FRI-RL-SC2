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

    count_of_actions = 8
    agent = Agent(PolicyValueModel(count_of_actions, 1936), optim, gamma, epsilon, 0.001, 0.5, 0.95, name, results_path, device)

    count_of_processes = 3
    count_of_envs = 2
    count_of_steps = 128
    batch_size = 128
    count_of_epochs = 4
    first_iteration = 0

    input_dim = (2, 84, 84)
    agent.train("", Env, count_of_actions, count_of_iterations, count_of_processes, count_of_envs, count_of_steps,
                count_of_epochs, batch_size, input_dim)


def animation():
    model = PolicyValueModel(8, 1936) # beacon
    model.to("cuda")
    model.load_state_dict(torch.load("results/beacon/models/best.pt"))

    env = Env(*"")
    i = 0
    while i < 50:
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
    #learning(301)
    animation()
