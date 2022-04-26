import torch
from agent.SACAgent import SACAgent
from environments.ant import Ant

env = Ant()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = SACAgent(env=env, device=device, learning_rate=0.01)

t = 0
while agent.winstreak < 10 and t < 250:
    t += 1
    agent.play_games(1, verbose=True)

agent.play_games(1, verbose=True, display_gameplay=True)
