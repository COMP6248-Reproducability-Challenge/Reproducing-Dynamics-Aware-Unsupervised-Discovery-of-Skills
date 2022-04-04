import gym

class Environment:
    def __init__(self):
        self.env = None
        self.observation = None
        self.reward = None
        self.done = False
        self.frames = 0
        self.won = False
        self.observation_space = None
        self.action_space = None
        self.pressed_action = None

    def reset_env(self):
        self.observation = self.env.reset()
        self.reward = None
        self.done = False
        self.frames = 0
        self.won = False
        self.lost = False
