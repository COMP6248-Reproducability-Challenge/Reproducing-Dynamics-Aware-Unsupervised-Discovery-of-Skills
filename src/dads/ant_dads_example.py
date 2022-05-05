import torch
from agent.DADSAgent import DADSAgent
from agent.SkillDynamics import SkillDynamics
from src.soft_actor_critic.agent.Actor import Actor
from src.soft_actor_critic.agent.Critic import Critic


from src.soft_actor_critic.environments.ant import Ant

env = Ant()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = DADSAgent(env=env, device=device, n_skills=4, learning_rate=3e-4)

# agent.save_models()
# agent.load_models()

t = 0
while t < 150:
    t += 1
    agent.play_games(1, verbose=False)
    agent.save_models()

print("done!")

agent.play_games(1, verbose=False, display_gameplay=True, train=False, skill=0)
agent.play_games(1, verbose=False, display_gameplay=True, train=False, skill=1)
agent.play_games(1, verbose=False, display_gameplay=True, train=False, skill=2)
agent.play_games(1, verbose=False, display_gameplay=True, train=False, skill=3)

