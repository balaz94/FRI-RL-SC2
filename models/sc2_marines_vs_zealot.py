import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.init import weights_init_orthogonal_head, weights_init_orthogonal_features
import torch.nn.init as init

class PolicyValueModel(nn.Module):
    def __init__(self, count_of_actions, features_size, init_features_model = None, init_policy_model = None, init_value_model = None):
        super(PolicyValueModel, self).__init__()

        #input 29x64x64
        self.features_model = nn.Sequential(
            nn.Conv2d(29, 32, 2, stride=2, padding=1), #32*32*32
            nn.ReLU(),
            nn.Conv2d(32, 64, 2, stride=2, padding=1), #64*16*16
            nn.ReLU(),
            nn.Conv2d(64, 64, 2, stride=2, padding=1), #64*8*8
            nn.ReLU(),
            nn.Conv2d(64, 32, 2, stride=2, padding=1), #64*4*4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(features_size, 128),  # feature size = 512
            nn.ReLU()
        )
        #count of actions = 32

        self.features_model.apply(
            weights_init_orthogonal_features if init_features_model is None else init_features_model)

        self.policy_model = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, count_of_actions) #mozno 27
        )
        self.policy_model.apply(weights_init_orthogonal_head if init_policy_model is None else init_policy_model)

        self.value_model = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.value_model.apply(weights_init_orthogonal_head if init_value_model is None else init_value_model)

    def forward(self, x):
        x = self.features_model(x)
        return self.policy_model(x), self.value_model(x)
