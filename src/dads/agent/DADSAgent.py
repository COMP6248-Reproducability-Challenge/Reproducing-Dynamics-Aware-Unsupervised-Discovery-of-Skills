import torch
from src.dads.agent.SkillDynamics import SkillDynamics
from src.soft_actor_critic.agent.SACAgent import SACAgent
from random import randint

class DADSAgent:
    def __init__(self, env, device, n_skills, agent_learning_rate=3e-4, dynamics_learning_rate=3e-4,
                 num_hidden_neurons=256, memory_length=1e6):
        self.n_skills = n_skills
        self.agent_learning_rate = agent_learning_rate
        self.dynamics_learning_rate = dynamics_learning_rate
        self.num_hidden_neurons = num_hidden_neurons
        self.memory_length = memory_length
        self.env = env
        self.state_shape = self.env.observation_space.shape[0]
        self.action_shape = self.env.action_space.shape[0]
        self.device = device
        self.memory_length = memory_length

        self.skills = {}
        for i in range(n_skills):
            agent = SACAgent(env=self.env, device=self.device, num_hidden_neurons=self.num_hidden_neurons,
                             writer=None, learning_rate=self.agent_learning_rate, alpha=0.1)
            dynamics = SkillDynamics(input_shape=self.state_shape, device=self.device, writer=None,
                                     num_hidden_neurons=self.num_hidden_neurons,
                                     learning_rate=self.dynamics_learning_rate)
            skill = {'policy': agent,
                     'dynamics': dynamics}
            self.skills[i] = skill

    def _sample_skill(self):
        return randint(0, len(self.skills)-1)

    def _choose_action(self, current_observation):
        pass

    def _calculate_intrinsic_reward(self, previous_obs, current_obs):
        pass

    def rollout(self, display_gameplay=False):
        skill_agent = self.skills[self._sample_skill()]['policy']
        skill_dynamics = self.skills[self._sample_skill()]['dynamics']
        self.env.reset_env()
        while not self.env.done:
            current_obs = torch.tensor(self.env.observation).reshape((1, -1)).to(self.device).type(torch.float)
            current_action = skill_agent.choose_action(current_obs)
            if display_gameplay:
                self.env.env.render()

            self.env.take_action(current_action.squeeze().cpu().numpy())
            next_obs = torch.tensor(self.env.observation).reshape((1, -1)).to(self.device).type(torch.float)
            done = self.env.done

            skill_agent.train_models(verbose=False)
            # TODO: train dynamics model here as well
            #  Question: Should the dynamics model be one neural network that takes the skill as input, or a separate
            #  NN per skill?

            reward = self._calculate_intrinsic_reward(previous_obs=current_obs, current_obs=next_obs)
            done = torch.tensor([[done]], dtype=torch.int).to(self.device)
            skill_agent.memory.append("observation", current_obs)
            skill_agent.memory.append("next_observation", next_obs)
            skill_agent.memory.append("action", current_action)
            skill_agent.memory.append("reward", reward)
            skill_agent.memory.append("done", done)