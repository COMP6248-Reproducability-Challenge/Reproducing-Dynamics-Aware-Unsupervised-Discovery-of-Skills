import torch
import torch.nn.functional as funcs
from torch.distributions.normal import Normal
from torch import nn
from torch import optim


class Actor(nn.Module):
    def __init__(self, learning_rate, input_shape, max_action, number_actions, device, writer, num_hidden_neurons):
        super(Actor, self).__init__()
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.max_action = max_action
        self.number_actions = number_actions
        self.num_hidden_neurons = num_hidden_neurons
        self.device = device
        self.writer = writer
        if self.writer is not None:
            self.training_epoch = 0

        # The reparameterisation trick: See paper: https://arxiv.org/pdf/1801.01290.pdf
        # Web discussion on reparameterisation: https://gregorygundersen.com/blog/2018/04/29/reparameterization/
        # Technical report on SAC policy loss (relevant): https://arxiv.org/pdf/2112.15568.pdf
        self.reparam_noise = 1e-6

        # As with the Critic, the paper uses 2 hidden layers with the same number of neurons.
        # The network then predicts the mu and log sigma parameters which define the normal distribution that will be
        # sampled from to get the action.
        # Log sigma is used rather than predicting sigma directly as it has better properties
        self.fc1 = nn.Linear(self.input_shape, self.num_hidden_neurons)
        self.fc2 = nn.Linear(self.num_hidden_neurons, self.num_hidden_neurons)
        self.mu = nn.Linear(self.num_hidden_neurons, self.number_actions)
        self.log_sigma = nn.Linear(self.num_hidden_neurons, self.number_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(self.device)

    def forward(self, observation):
        out = self.fc1(observation)
        out = funcs.relu(out)
        out = self.fc2(out)
        out = funcs.relu(out)
        mu = self.mu(out)
        log_sigma = self.log_sigma(out)
        # Clamping the variance of the sampling distribution makes the learning process more stable as it avoids
        # a covariance so large that the action we sample from the distribution is essentially a sample from a uniform
        # distribution. Common choices in other online implementations are -20 and 2:
        log_sigma_minimum = -20
        log_sigma_maximum = 2
        log_sigma = torch.clamp(log_sigma, min=log_sigma_minimum, max=log_sigma_maximum)
        return mu, log_sigma

    def sample_normal(self, observation, reparameterise):
        if reparameterise:
            pred_mu, pred_log_sigma = self.forward(observation)
            pred_sigma = torch.exp(pred_log_sigma)
            normal = Normal(pred_mu, pred_sigma)
            actions_base = normal.rsample()  # rsample() is sampling by applying the reparameterisation trick.
        else:
            with torch.no_grad():
                pred_mu, pred_log_sigma = self.forward(observation)
                pred_sigma = torch.exp(pred_log_sigma)
                normal = Normal(pred_mu, pred_sigma)
                actions_base = normal.sample()
        # To get the action, we use tanh to 'clamp' the action bounds to be +-1 - scale it to the action bounds later
        tanh_actions = torch.tanh(actions_base).to(self.device)

        # The lob probabilities are used in the policy loss, as well as the entropy part of the q learning loss.
        log_probs = normal.log_prob(actions_base) - torch.log(1 - tanh_actions.pow(2) + self.reparam_noise)
        # sum across the log probabilities as we have assumed that each action is an independent normal, so
        # the sum of the log probs across the columns is the log probs of the 8-dimensional action
        log_probs = log_probs.sum(1, keepdim=True)

        return tanh_actions, log_probs
