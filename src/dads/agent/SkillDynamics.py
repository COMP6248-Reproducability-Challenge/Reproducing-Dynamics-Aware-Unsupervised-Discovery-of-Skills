import torch
import torch.nn.functional as funcs
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from torch import nn
from torch import optim


class SkillDynamics(nn.Module):
    def __init__(self, input_shape, output_shape, device, num_hidden_neurons=256, learning_rate=3e-4):
        super(SkillDynamics, self).__init__()
        self.input_shape = input_shape  # This includes the skill encoder
        self.output_shape = output_shape  # This is just the size of the state space
        self.device = device
        self.num_hidden_neurons = num_hidden_neurons  # Paper uses same number across layers
        self.learning_rate = learning_rate
        # The first layer takes the state input plus a one-hot encoding of which skill is active
        self.batchnorm = nn.BatchNorm1d(num_features=self.num_hidden_neurons)
        self.fc1 = nn.Linear(self.input_shape, self.num_hidden_neurons)
        self.fc2 = nn.Linear(self.num_hidden_neurons, self.num_hidden_neurons)
        # Each expert is a multinomial gaussian, with the same dimension as the input state space
        self.expert1_mu = nn.Linear(self.num_hidden_neurons, output_shape)
        self.expert2_mu = nn.Linear(self.num_hidden_neurons, output_shape)
        self.expert3_mu = nn.Linear(self.num_hidden_neurons, output_shape)
        self.expert4_mu = nn.Linear(self.num_hidden_neurons, output_shape)
        # A softmax layer is used as part of gating model to decide when to use which expert. This is currently
        # implemented a Linear layer, but we should confirm this is the same as in the paper.
        # The softmax acts to choose which expert to use, there's 4 hardcoded experts, so 4 output neurons:
        self.softmax_input = nn.Linear(self.num_hidden_neurons, 4)
        # All experts use an identity matrix as the standard deviation.
        self.sigma = torch.eye(self.output_shape, requires_grad=False, device=self.device)  # We don't want to update this so no gradients
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(self.device)

    def forward(self, observation_and_skill):
        out = self.fc1(observation_and_skill)
        out = self.batchnorm(out)
        out = funcs.relu(out)
        out = self.fc2(out)
        out = self.batchnorm(out)
        out = funcs.relu(out)
        mu1 = self.expert1_mu(out)
        mu2 = self.expert2_mu(out)
        mu3 = self.expert3_mu(out)
        mu4 = self.expert4_mu(out)
        softmax_gate = funcs.softmax(self.softmax_input(out), dim=1)
        return mu1, mu2, mu3, mu4, softmax_gate

    def _create_skill_encoded(self, skill_int):
        skill_one_hot_encoder = torch.zeros(self.n_skills, dtype=torch.float, device=self.device)
        skill_one_hot_encoder[skill_int] = 1.0
        skill_one_hot_encoder = skill_one_hot_encoder.reshape((1, -1))
        return skill_one_hot_encoder

    def sample_next_state(self, state_and_skill, reparam=False):
        mu1, mu2, mu3, mu4, softmax_gate = self.forward(state_and_skill)

        combined_mus = (mu1.T * softmax_gate[:, 0]).T + \
                       (mu2.T * softmax_gate[:, 1]).T + \
                       (mu3.T * softmax_gate[:, 2]).T + \
                       (mu4.T * softmax_gate[:, 3]).T

        # rsample() uses reparameterisation trick, so that the gradients can flow backwards through this sampling step
        next_state_distribution = MultivariateNormal(combined_mus, self.sigma)
        if reparam:
            delta = next_state_distribution.rsample()
        else:
            delta = next_state_distribution.sample()
        next_state = state_and_skill[:, 0:2] + delta
        return next_state, delta, next_state_distribution

    def train_model(self, previous_obs_and_skill, current_obs_and_skill, verbose=False):
        # Remember: The dynamics model predicts the delta from the current state to the next
        self.optimizer.zero_grad()
        _, predicted_delta, distribution = self.sample_next_state(previous_obs_and_skill, reparam=True)
        actual_delta = (current_obs_and_skill - previous_obs_and_skill)[:, 0:2]
        actual_delta_scaled = ((actual_delta - torch.mean(actual_delta, dim=0)) / torch.std(actual_delta, dim=0))
        loss = torch.mean(-1.0 * distribution.log_prob(actual_delta))
        if verbose:
            print("Skill dynamics loss: ", loss.item())
        loss.backward()
        self.optimizer.step()
