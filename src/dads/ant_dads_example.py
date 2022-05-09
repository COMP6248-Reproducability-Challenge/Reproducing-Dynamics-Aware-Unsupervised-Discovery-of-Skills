import torch
from agent.DADSAgent import DADSAgent
from agent.SkillDynamics import SkillDynamics
from src.soft_actor_critic.agent.Actor import Actor
from src.soft_actor_critic.agent.Critic import Critic
from torch import optim
from src.dads.environments.ant_truncated import Ant_Truncated_State
from datetime import datetime

env = Ant_Truncated_State()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = DADSAgent(env=env, device=device, n_skills=4, learning_rate=3e-4)

# agent.save_models()
agent.load_models()
#
# learning_rate = 0.01 # 3e-4
# agent.skill_dynamics.optimizer = optim.Adam(agent.skill_dynamics.parameters(), lr=learning_rate)
# agent.actor.optimizer = optim.Adam(agent.actor.parameters(), lr=learning_rate)
# agent.critic1.optimizer = optim.Adam(agent.critic1.parameters(), lr=learning_rate)
# agent.critic2.optimizer = optim.Adam(agent.critic2.parameters(), lr=learning_rate)
# agent.critic_target1.optimizer = optim.Adam(agent.critic_target1.parameters(), lr=learning_rate)
# agent.critic_target2.optimizer = optim.Adam(agent.critic_target2.parameters(), lr=learning_rate)


t = 0
while True:
    t += 1
    agent.play_games(1, verbose=False)
    agent.save_models()
    with open("iterations.txt", "a") as iter_file:
        iter_file.seek(0)
        iter_file.write("iter {} at time: {}\n".format(t, datetime.now(tz=None)))

print("done!")

agent.play_games(1, verbose=False, display_gameplay=True, train=False, skill=0)
agent.play_games(1, verbose=False, display_gameplay=True, train=False, skill=1)
agent.play_games(1, verbose=False, display_gameplay=True, train=False, skill=2)
agent.play_games(1, verbose=False, display_gameplay=True, train=False, skill=3)

