# here I will share our NN model write by PyTorch
# for this model our input is 1600 feature: 200 original var, 200 count, 200 count * var, 200 var ./ count
# 200 value rank groupby different count, 200 zscore ((var - mean)/ std) groupby count
# and 200 value mask the count=1 (replace value that has count = 1 with -100), and also zscore for count=1 and count!=1

# input is the original feature, no scale transfer

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.modules.module import Module

class Net(nn.Module):
    def __init__(self, input_dim=1600, hidden_dim=256, dropout=0.5):
        super(Net, self).__init__()

        self.inpt_dim = input_dim
        self.hidden_dim = hidden_dim
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv1 = ConvReluBn1d(1, 64, 3)
        self.conv2 = ConvReluBn1d(64, 128, 3)
        self.conv3 = ConvReluBn1d(128, 64, 3)
        self.conv4 = ConvReluBn1d(64, 8, 3)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.name = 'bn1600-conv1d3x64relubn-conv1d3x128relubn-conv1d3x64relubn-conv1d3x6relubn-256relubn-dp0p5-1'
        self.bn = nn.BatchNorm1d(input_dim, momentum=0.5)
        conv_dim = 12736 # just hard code the output dimension for convenience
        self.fc3 = nn.Linear(conv_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = self.bn(x)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.dropout(x)
        logit = self.classifier(x)
        logit = logit.squeeze(1)
        return logit
        
class ConvReluBn1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=None):
        super(ConvReluBn1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.bn(x)

        if not self.dropout is None:
            x = self.dropout(x)

        return x