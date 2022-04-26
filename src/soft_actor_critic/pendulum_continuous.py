import torch
from agent.SACAgent import SACAgent
from environments.pendulum import Pendulum

env = Pendulum()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = SACAgent(env=env, device=device)

t = 0
while agent.winstreak < 10 and t < 150:
    t += 1
    agent.play_games(1, verbose=True)

agent.play_games(1, verbose=True, display_gameplay=True)
