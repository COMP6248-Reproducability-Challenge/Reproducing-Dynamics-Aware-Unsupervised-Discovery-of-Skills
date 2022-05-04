import torch
from src.soft_actor_critic.agent.Memory import Memory

class SkillDynamicsMemory(Memory):
    def __init__(self, memory_length, device):
        self.memory_length = memory_length
        self.device = device
        super(SkillDynamicsMemory, self).__init__(memory_length=self.memory_length, device=self.device)
        self.skills = torch.tensor([], dtype=torch.int, device=self.device)

    def sample_memory(self):
        skills = self.skills
        obs = self.observation
        next_obs = self.next_observation
        actions = self.action
        rewards = self.reward
        done = self.done
        return skills, obs, next_obs, actions, rewards, done

    def wipe(self):
        super(SkillDynamicsMemory, self).__init__(memory_length=self.memory_length, device=self.device)
        self.skills = torch.tensor([], dtype=torch.int, device=self.device, requires_grad=False)
