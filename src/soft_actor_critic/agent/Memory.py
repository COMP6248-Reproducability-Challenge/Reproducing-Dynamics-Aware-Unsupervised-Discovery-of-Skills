import torch
import numpy as np


class Memory:
    def __init__(self, memory_length, device):
        self.memory_length = memory_length
        self.device = device
        self.observation = torch.tensor([]).to(self.device)
        self.next_observation = torch.tensor([]).to(self.device)
        self.action = torch.tensor([]).to(self.device)
        self.reward = torch.tensor([], dtype=torch.int).to(self.device)
        self.done = torch.tensor([], dtype=torch.bool).to(self.device)

    def append(self, mem_type, data):
        mem = getattr(self, mem_type)
        if len(mem) < self.memory_length:
            mem = torch.cat((mem, data), dim=0)
        else:
            mem = torch.cat((mem[1:], data))
        setattr(self, mem_type, mem)

    def sample_memory(self, sample_length):
        memory_size = len(self.observation)
        sample_index = list(np.random.choice(range(memory_size), sample_length, replace=True))
        obs = self.observation[sample_index]
        next_obs = self.next_observation[sample_index]
        actions = self.action[sample_index]
        rewards = self.reward[sample_index]
        done = self.done[sample_index]
        return obs, next_obs, actions, rewards, done

    def wipe(self):
        self.observation = torch.tensor([]).to(self.device)
        self.next_observation = torch.tensor([]).to(self.device)
        self.action = torch.tensor([]).to(self.device)
        self.reward = torch.tensor([]).to(self.device)
        self.done = torch.tensor([]).to(self.device)
