import torch
import torch.nn.functional as funcs
from torch import nn
from torch import optim


class SkillDynamics(nn.Module):
    def __init__(self, input_shape, device, writer=None, num_hidden_neurons = 256, learning_rate=3e-4):
        super(SkillDynamics, self).__init__()
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.device = device
        self.writer = writer
        self.num_hidden_neurons = num_hidden_neurons  # Paper uses same number across layers
        if self.writer is not None:
            self.training_epoch = 0

        self.fc1 = nn.Linear(self.input_shape, self.num_hidden_neurons)
        self.fc2 = nn.Linear(self.num_hidden_neurons, self.num_hidden_neurons)
        # Each expert is a multinomial gaussian, with the same dimension as the input state space
        self.expert1_mu = nn.Linear(self.num_hidden_neurons, input_shape)
        self.expert2_mu = nn.Linear(self.num_hidden_neurons, input_shape)
        self.expert3_mu = nn.Linear(self.num_hidden_neurons, input_shape)
        self.expert4_mu = nn.Linear(self.num_hidden_neurons, input_shape)
        # A softmax layer is used as part of gating model to decide when to use which expert. This is currently
        # implemented a Linear layer, but we should confirm this is the same as in the paper.
        # The softmax acts to choose which expert to use, there's 4 hardcoded experts, so 4 output neurons:
        self.softmax_input = nn.Linear(self.num_hidden_neurons, 4)
        # All experts use an identity matrix as the standard deviation.
        self.sigma = torch.eye(self.input_shape)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(self.device)

    def forward(self, observation):
        """
        The observation is used to predict a Gaussian mixture-of-experts' predictions of the means of the next state.
        The standard deviation is assumed by the paper to be the identity matrix across the experts.

        This method was built from the paper's description of their implementation:

        We input the current state space. The output of the hidden layers is used as an input to the mixture of
        experts which is fixed to be four. Each expert is a gaussian distribution, with the input to the experts
        linearly transformed to output the parameters of the gaussian distribution, and a discrete distribution over
        the experts using a softmax distribution.
        "In practice, we fix the covariance matrix of the gaussian experts to be the identity matrix, so we only need
        to output the means for the experts. We use batch normalisation for both input and the hidden layers. We
        normalise the output targets using their batch average and batch standard deviation, similar to batch
        normalisation"

        :param observation: The current state s
        :return: The four expert's means plus the softmax layer's output, which is the choice of expert
        """

        out = self.fc1(observation, dim=1)
        out = funcs.relu(out)
        out = self.fc2(out)
        out = funcs.relu(out)
        # TODO: replace this implementation with the MixtureSameFamily class in PyTorch.
        mu1 = self.expert1_mu(out)
        mu2 = self.expert2_mu(out)
        mu3 = self.expert3_mu(out)
        mu4 = self.expert4_mu(out)
        softmax_gate = funcs.softmax(self.softmax_input(out))
        return mu1, mu2, mu3, mu4, softmax_gate

    def sample_next_state(self, mu1, mu2, mu3, mu4, softmax_gate):
        """
        Returns a prediction of the next state when provided the mixture-of-experts' predictions.

        :param mu1: The predicted mean of the next state distribution from expert 1
        :param mu2: The predicted mean of the next state distribution from expert 2
        :param mu3: The predicted mean of the next state distribution from expert 3
        :param mu4: The predicted mean of the next state distribution from expert 4
        :param softmax_gate: The output from the softmax gating model used to stochastically choose which expert to use
        :return: The next state, sampled from the stochastically chosen expert for this state.
        """
        mus = [mu1, mu2, mu3, mu4]
        # We stochastically decide which mu to use based on the softmax_gate's as probabilities:
        chosen_expert = torch.multinomial(softmax_gate, 1)
        expert_mu = mus[chosen_expert]
        next_state = torch.normal(expert_mu, self.sigma)
        return next_state

    def train_model(self, previous_obs, current_obs):
        # TODO: Assuming that memory is sampled and passed in, we
        #  then need to use batch normalisation while we're training, both for the input and the hidden layers as well.
        # Remember: The dynamics model predicts the delta from the current state to the next
        predicted_observation_delta = self.forward(previous_obs)
        actual_observation_delta = current_obs - previous_obs

        self.optimizer.zero_grad()
        loss = torch.nn.MSELoss(predicted_observation_delta, actual_observation_delta)
        loss.backward()
        self.optimizer.step()
