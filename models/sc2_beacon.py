import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.init import weights_init_orthogonal_head, weights_init_orthogonal_features

class PolicyValueModel(nn.Module):
    def __init__(self, count_of_actions, features_size, init_features_model = None, init_policy_model = None, init_value_model = None):
        super(PolicyValueModel, self).__init__()

        #input 2x84x84
        self.features_model = nn.Sequential(
            nn.Conv2d(2, 8, 3, stride = 2, padding = 1),    #8x42x42
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride = 2, padding = 1),   #16x21x21
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride = 2, padding = 1),  #16x11x11
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(features_size, 128),  # feature size = 1936
            nn.ReLU()
        )
        self.features_model.apply(weights_init_orthogonal_features if init_features_model is None else init_features_model)

        self.policy_model = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, count_of_actions)
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
