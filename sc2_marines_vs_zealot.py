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
    lr = 0.00025
    name = 'ppo'
    optim = 'Adam'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_path = 'results/marines_vs_zealot/'

    actions = 27

    #agent = PolicyValueModel(0.99, actions, Net(), 0.001, beta_entropy=0.01, id=0, name='sc2_custom_ZealotArmorUp_re_mc_10_ntr', depth = 21, height = 64, width = 64)
    agent = Agent(PolicyValueModel(27, 800), gamma, entropy_loss_coef, value_loss_coef ,epsilon, lr, name, optim, device,results_path)
    #agent.model.load_state_dict(torch.load('models/sc2_custom_ZealotArmorUp_onlytest_0_2500_ppo.pt'))

    # run_id = int(datetime.datetime.timestamp(datetime.datetime.now()))
    # os.mkdir("logs/sc2/custom_ZealotArmorUp/" + str(run_id))
    # total_time_file = open("logs/sc2/custom_ZealotArmorUp/" + str(run_id) + "/Train_time.txt", "w", buffering=1)
    # START_TIME = datetime.datetime.now()
    # signal.signal(signal.SIGTERM, at_exist)
    # signal.signal(signal.SIGINT, at_exist)

    count_of_processes = 3
    count_of_envs = 2
    count_of_steps = 1024
    batch_size = 1024
    count_of_epochs = 4
    first_iteration = 0
    input_dim = (21, 64, 64)

    agent.train("", Env, count_of_processes, count_of_envs, count_of_iterations, count_of_steps, batch_size, count_of_epochs, first_iteration, input_dim)

   # workers = []
    #for id in range(10):
     #   env = Env()
     #   w = Worker(id, env, agent)
     #   workers.append(w)

    #agent.learn(workers, 764, 10001, 764, batches_count = 8, max_mc = 10)

    #total_time_file.close()

# def animation():
#     scores = []
#     avg_scores = deque([])
#     actions = 27
#     env = Env()
#
#     agent = AgentPPO(0.99, actions, Net(), 0.001, beta_entropy=0.01, id=0, name='sc2_custom_ZealotArmorUp_re_mc_10_ntr', depth = 21, height = 64, width = 64)
#     agent.model.load_state_dict(torch.load('models/sc2_custom_ZealotArmorUp_re_mc_10_ntr_0_ppo.pt'))
#
#     i = 0
#     while True:
#         score = 0
#         terminal = False
#         state = env.reset()
#         i += 1
#
#         while not terminal:
#             time.sleep(0.25)
#             action = agent.choose_action(torch.from_numpy(state).double())
#             state, reward, terminal, _ = env.step(action)
#             score += reward
#
#         avg_scores.append(score)
#         if len(avg_scores) > MAX_SCORE_COUNT:
#             avg_scores.popleft()
#
#         scores.append(score)
#         print('episode: ', i, '\t\tavg-100 score:',
#               round(np.average(avg_scores), 3))


# def at_exist(signum, frame):
#     global score_file
#     global steps_file
#     global total_time_file
#     global START_TIME
#     global agent
#
#     total_time_file.write("Train time: " + str(datetime.datetime.now() - START_TIME) + '\n')
#     total_time_file.write("Episodes: " + str(agent.get_episodes()) + '\n')
#     total_time_file.close()
#     score_file.flush()
#     steps_file.flush()
#     score_file.close()
#     steps_file.close()



# class Stat:
#     def __init__(self, score_file, steps_file):
#         self.score_file = score_file
#         self.steps_file = steps_file
#         self.i = 0
#
#     def save_stat(self, env, score):
#         self.i += 1
#         self.score_file.write(str(score) + '\n')
#         self.steps_file.write(str(env.get_raw_obs().observation.game_loop[0]) + '\n')

if __name__ == '__main__':
    learning(2000)
    #animation()
