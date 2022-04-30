import gym
from .environment import Environment

class MountainCarContinuous(Environment):
    def __init__(self):
        super().__init__()
        self.env = gym.make('MountainCarContinuous-v0')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def take_action(self, action):
        obs, reward, done, _ = self.env.step([action.item()])
        self.observation = obs
        self.reward = reward
        self.done = done
        self.frames += 1
        if self.done and self.frames < 999:
            self.won = True
        else:
            self.won = False

    def reset_env(self):
        self.observation = self.env.reset()
        self.reward = None
        self.done = False
        self.frames = 0
        self.won = False
        self.lost = False
