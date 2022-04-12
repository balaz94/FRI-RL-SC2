from collections import deque

from pyglet.resource import animation

from envs.sc2_shards_env import Env
from models.sc2_shards import PolicyValueModel
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import gym
import graph.make_graph as graph
from agents.ppo_parallel import Agent


MAX_SCORE_COUNT = 100

def learning(count_of_iterations):
    gamma = 0.99
    entropy_loss_coef = 0.001
    value_loss_coef = 0.5
    epsilon = 0.1
    
    lr = 0.00025
    name = 'shards'
    optim = 'Adam'
    device = "cuda"
    results_path = 'results/shards/'

    count_of_actions = 16
    agent = Agent(PolicyValueModel(count_of_actions, 512), optim, gamma, epsilon, 0.001, 0.5, 0.95, name, results_path, device)

    count_of_processes = 3
    count_of_envs = 2
    count_of_steps = 128
    batch_size = 128
    count_of_epochs = 4

    first_iteration = 0
    input_dim = (3, 64, 64)
    agent.load_model("results/shards/models/best.pt")
    agent.train("", Env, count_of_actions, count_of_iterations, count_of_processes, count_of_envs, count_of_steps,
                count_of_epochs, batch_size, input_dim)

def animation():
    model = PolicyValueModel(16, 512) # shards
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

class VideoRecorder(gym.Wrapper):
    def __init__(self, env, file_name="video.avi"):
        super(VideoRecorder, self).__init__(env)

        self.height = 2 * 64
        self.width = 2 * 64
        env.get_state_from_obs()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(file_name, fourcc, 50.0, (self.width, self.height))

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        im_bgr = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)

        resized = cv2.resize(im_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)

        self.writer.write(resized)

        return state, reward, done, info

    def reset(self):
        return self.env.reset()


def makegraph():
    graph.scatter_plot("results/zergs/data/zergs.txt")

if __name__ == "__main__":
   #learning(11)
   #animation()
   makegraph()

