import torch
import torch.nn.functional as funcs
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch import nn
from torch import optim


class SkillDynamics(nn.Module):
    def __init__(self, input_shape, output_shape, device, num_hidden_neurons=256, learning_rate=3e-4):
        super(SkillDynamics, self).__init__()
        self.input_shape = input_shape  # This includes the skill encoder
        self.output_shape = output_shape  # This is just the size of the state space
        self.device = device
        self.num_hidden_neurons = num_hidden_neurons  # Paper uses same number across layers and networks
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
        # # All experts use an identity matrix as the standard deviation.
        self.sigma = torch.eye(self.output_shape, requires_grad=False, device=self.device)
        # We don't want to update this so no gradients
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

    def sample_next_state(self, state_and_skill, reparam=False):
        mu1, mu2, mu3, mu4, softmax_gate = self.forward(state_and_skill)
        _, top_gate_ind = torch.topk(softmax_gate, 1)

        # Here we take ith expert's prediction of the mean multiplied by the softmax value IF the ith softmax was
        # the maximum. The other mus are "added" but are zero because they are multiplied by False.
        # TODO: check whether this allows the gradients to propogate backwards. It might not?
        expert_mu = (mu1.T * softmax_gate[:, 0]).T * (top_gate_ind == 0) + \
                       (mu2.T * softmax_gate[:, 1]).T * (top_gate_ind == 1) + \
                       (mu3.T * softmax_gate[:, 2]).T * (top_gate_ind == 2) + \
                       (mu4.T * softmax_gate[:, 3]).T * (top_gate_ind == 3)

        # rsample() uses reparameterisation trick, so that the gradients can flow backwards through the parameters
        # in contrast to sample() which blocks gradients (as it's a random sample)
        next_state_distribution = MultivariateNormal(expert_mu, self.sigma)
        if reparam:
            delta = next_state_distribution.rsample()
        else:
            delta = next_state_distribution.sample()
        # TODO: fix this to be general, as 0:2 only relates to the mountain car environment's dimension of 2
        next_state = state_and_skill[:, 0:2] + delta # next_state = state_and_skill[:, 0:2] + delta
        return next_state, delta, next_state_distribution

    def train_model(self, previous_obs_and_skill, current_obs_and_skill, verbose=False):
        # Remember: The dynamics model predicts the delta from the current state to the next
        self.optimizer.zero_grad()
        _, _, distribution = self.sample_next_state(previous_obs_and_skill, reparam=True)
        # TODO: fix this indexing to be general, see TODO on line 71 above
        # We index after this difference as we only want  delta of the state and not the space-skill concatenation
        actual_delta = (current_obs_and_skill - previous_obs_and_skill)[:, 0:2]
        # TODO: check whether the scaling performed on the targets below gives the correct values?
        actual_delta_scaled = ((actual_delta - torch.mean(actual_delta, dim=0)) / torch.std(actual_delta, dim=0))
        loss = -1.0 * torch.mean(distribution.log_prob(actual_delta_scaled))
        loss.backward()
        if verbose:
            print("Skill dynamics loss: ", loss.item())
        self.optimizer.step()
