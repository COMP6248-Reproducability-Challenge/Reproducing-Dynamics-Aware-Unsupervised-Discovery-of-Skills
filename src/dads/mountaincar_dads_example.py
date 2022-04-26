import torch
from agent.DADSAgent import DADSAgent
from src.soft_actor_critic.environments.mountaincar_cont import MountainCarContinuous

env = MountainCarContinuous()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = DADSAgent(env=env, device=device, n_skills=2)

print("done!")