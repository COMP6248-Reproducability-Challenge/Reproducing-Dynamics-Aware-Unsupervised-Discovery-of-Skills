import torch
import torch.nn.functional as funcs
from torch import nn
from torch import optim


class SkillDynamics(nn.Module):
    def __init__(self, learning_rate, input_shape, device, writer, num_hidden_neurons):
        super(SkillDynamics).__init__()
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
        # TODO: This softmax bit may be wrong, it's half guesswork. A softmax layer is used as part of gating model to
        #  decide when to use which expert. I'm choosing to implement this as another Linear layer but we should
        #  confirm this is the same as in the paper.
        # The softmax acts to choose which expert to use, there's 4 hardcoded experts:
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
        :return: The four expert's means plus the softmax layer output
        """

        out = self.fc1(observation, dim=1)
        out = funcs.relu(out)
        out = self.fc2(out)
        out = funcs.relu(out)
        mu1 = self.expert1_mu(out)
        mu2 = self.expert2_mu(out)
        mu3 = self.expert3_mu(out)
        mu4 = self.expert4_mu(out)
        softmax_gate = funcs.softmax(self.softmax_input(out))
        return mu1, mu2, mu3, mu4, softmax_gate

    def predict_next_state(self, mu1, mu2, mu3, mu4, softmax_gate):
        pass

