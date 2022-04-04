import torch
import torch.nn.functional as funcs
from torch import nn
from torch import optim


class Critic(nn.Module):
    def __init__(self, learning_rate, input_shape, number_actions, device, writer, num_hidden_neurons):
        super(Critic, self).__init__()
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.number_actions = number_actions
        self.num_hidden_neurons = num_hidden_neurons
        self.device = device
        self.writer = writer
        if self.writer is not None:
            self.training_epoch = 0

        # We input the current state and the selected action to get a prediction of the Q value for the state-action
        # combination. The DADS paper uses 2 hidden layers with the same number of neurons in each, so
        # num_hidden_neurons is the number of hidden neurons in both layers for simplicity.
        # Output layer is a single neuron predicting Q value.
        self.fc1 = nn.Linear(self.input_shape + self.number_actions, self.num_hidden_neurons)
        self.fc2 = nn.Linear(self.num_hidden_neurons, self.num_hidden_neurons)
        self.fc3 = nn.Linear(self.num_hidden_neurons, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(self.device)

    def forward(self, observation, action):
        # From state-action pair, predict the (scalar) Q value.
        # No (or linear) activation layer as we're predicting Q which is an unbounded real value
        out = self.fc1(torch.cat([observation, action], dim=1))
        out = funcs.relu(out)
        out = self.fc2(out)
        out = funcs.relu(out)
        out = self.fc3(out)
        return out
