import gym
from src.soft_actor_critic.environments.environment import Environment
import numpy as np
from gym.spaces import Box

class Ant_Truncated_State(Environment):
    def __init__(self):
        super().__init__()
        self.env = gym.make('Ant-v3')
        # self.observation_space = self.env.observation_space
        self.observation_space = Box(-float('inf'), float('inf'), shape=(27,))
        self.action_space = self.env.action_space
        self.xy_coords = self.env.sim.data.qpos.flat[:2]

    def take_action(self, action):
        _, reward, done, _ = self.env.step(action)
        # cutting out the parts of the state space that the DADS paper similarly removed
        obs = np.concatenate([
          self.env.sim.data.qpos.flat[2:15],
          self.env.sim.data.qvel.flat[:14], ])
        self.xy_coords = list(self.env.sim.data.qpos.flat[:2])
        self.observation = list(obs)
        self.reward = reward
        self.done = done
        self.frames += 1
        if self.done and self.frames > 1000:
            self.won = True
        else:
            self.won = False

    def reset_env(self):
        self.env.reset()
        self.observation = np.concatenate([
          self.env.sim.data.qpos.flat[2:15],
          self.env.sim.data.qvel.flat[:14], ])
        self.reward = None
        self.done = False
        self.frames = 0
        self.won = False
        self.lost = False
