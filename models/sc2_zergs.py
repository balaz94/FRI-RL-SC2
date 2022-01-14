import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.init import weights_init_orthogonal_head, weights_init_orthogonal_features

class PolicyValueModel(nn.Module):
    def __init__(self, count_of_actions, features_size, init_features_model = None, init_policy_model = None, init_value_model = None):
        super(PolicyValueModel, self).__init__()

        #input 3x64x64
        self.features_model = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride = 2, padding = 1),    #32x32x32
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride = 2, padding = 1),   #32x16x16, Hlbka - Sirka
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride = 2, padding = 1),  #32x8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(features_size, 256),  # feature size = 2048
            nn.ReLU()
        )
        self.features_model.apply(weights_init_orthogonal_features if init_features_model is None else init_features_model)

        self.policy_model = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, count_of_actions)
        )
        self.policy_model.apply(weights_init_orthogonal_head if init_policy_model is None else init_policy_model)

        self.value_model = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.value_model.apply(weights_init_orthogonal_head if init_value_model is None else init_value_model)

    def forward(self, x):
        x = self.features_model(x)
        return self.policy_model(x), self.value_model(x)
