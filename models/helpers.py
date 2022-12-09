import torch

import pandas as pd
import numpy as np
import torch.nn as nn
import xgboost as xgb


class NN(nn.Module):
    def __init__(self, input_size, n_hidden_neurons):
        super(NN, self).__init__()

        self.fc1 = nn.Linear(input_size, n_hidden_neurons[0])
        self.act1 = torch.nn.ReLU()
        self.fc2 = nn.Linear(n_hidden_neurons[0], 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x.reshape(-1)
