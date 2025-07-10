import torch
import torch.nn as nn

"""
EventNet is a neural network designed to process event level features into a event representation. That is later feed into classifier along with object level representation
"""
class EventNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EventNet, self).__init__() # initialize the parent class nn.Module
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Forward pass through the network
        :param x: Input tensor of shape (batch_size, input_dim)
        :return: Output tensor of shape (batch_size, output_dim)
        """
        return self.net(x) # this returns the event representation in 64 dimensional embedding space