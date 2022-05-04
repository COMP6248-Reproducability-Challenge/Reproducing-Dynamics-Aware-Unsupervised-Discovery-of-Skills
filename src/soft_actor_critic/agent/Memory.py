import torch
from random import choices

class Memory:
    def __init__(self, memory_length, device):
        self.memory_length = memory_length
        self.device = device
        self.observation = torch.tensor([], device=self.device, requires_grad=False)
        self.next_observation = torch.tensor([], device=self.device, requires_grad=False)
        self.action = torch.tensor([], device=self.device, requires_grad=False)
        self.reward = torch.tensor([], dtype=torch.int, device=self.device, requires_grad=False)
        self.done = torch.tensor([], dtype=torch.bool, device=self.device, requires_grad=False)

    def append(self, mem_type, data):
        mem = getattr(self, mem_type)
        if len(mem) < self.memory_length:
            mem = torch.cat((mem, data), dim=0)
        else:
            mem = torch.cat((mem[1:], data))
        setattr(self, mem_type, mem)

    def sample_memory(self, sample_length):
        memory_size = len(self.observation)
        sample_index = choices(range(memory_size), k=sample_length)
        obs = self.observation[sample_index]
        next_obs = self.next_observation[sample_index]
        actions = self.action[sample_index]
        rewards = self.reward[sample_index]
        done = self.done[sample_index]
        return obs, next_obs, actions, rewards, done