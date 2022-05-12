import torch
import torch.nn.functional as funcs
from src.soft_actor_critic.agent.Actor import Actor
from src.soft_actor_critic.agent.Critic import Critic
from src.soft_actor_critic.agent.Memory import Memory


class SACAgent:
    def __init__(self, env, device, num_hidden_neurons=256, writer=None, learning_rate=3e-4, discount_rate=0.99,
                 memory_length=1000000, batch_size=256, polyak=0.995, alpha=0.1, policy_train_delay_modulus=2,
                 input_shape=None):
        self.device = device
        self.learning_rate = learning_rate
        self.env = env
        self.input_shape = self.env.observation_space.shape[0] if input_shape is None else input_shape
        self.output_shape = self.env.action_space.shape[0]
        self.action_scale = torch.FloatTensor(
                (self.env.action_space.high - self.env.action_space.low) / 2.).to(self.device)
        self.discount_rate = discount_rate
        self.memory_length = memory_length
        self.num_hidden_neurons = num_hidden_neurons
        self.batch_size = batch_size
        self.writer = writer
        self.polyak = polyak
        self.policy_train_delay_modulus = policy_train_delay_modulus
        self.updates = 0
        self.alpha = torch.tensor([alpha]).to(self.device)

        self.actor = Actor(learning_rate=self.learning_rate, input_shape=self.input_shape, max_action=self.action_scale,
                           number_actions=self.output_shape, device=self.device, writer=self.writer,
                           num_hidden_neurons=self.num_hidden_neurons)
        self.critic1 = Critic(learning_rate=self.learning_rate, input_shape=self.input_shape,
                              number_actions=self.output_shape, device=self.device, writer=self.writer,
                              num_hidden_neurons=self.num_hidden_neurons)
        self.critic2 = Critic(learning_rate=self.learning_rate, input_shape=self.input_shape,
                              number_actions=self.output_shape, device=self.device, writer=self.writer,
                              num_hidden_neurons=self.num_hidden_neurons)
        self.critic_target1 = Critic(learning_rate=self.learning_rate, input_shape=self.input_shape,
                                     number_actions=self.output_shape, device=self.device, writer=self.writer,
                                     num_hidden_neurons=self.num_hidden_neurons)
        self.critic_target2 = Critic(learning_rate=self.learning_rate, input_shape=self.input_shape,
                                     number_actions=self.output_shape, device=self.device, writer=self.writer,
                                     num_hidden_neurons=self.num_hidden_neurons)

        self.memory = Memory(memory_length=self.memory_length, device=self.device)

        self.initial_target_update(self.critic_target1, self.critic1)
        self.initial_target_update(self.critic_target2, self.critic2)

        # Finally, some playing logic:
        self.episode_counter = 0
        self.winstreak = 0
        self.total_wins = 0

    def choose_action(self, observation):
        # Because we aren't using the gradients here, we don't need to do the reparameterisation trick, and don't need
        # to store the log probs:
        actions, _ = self.actor.sample_normal(observation=observation, reparameterise=False)
        actions = self.rescale_action(actions)
        return actions

    def rescale_action(self, action):
        return action * self.action_scale

    def train_models(self, verbose):
        if len(self.memory.observation) < self.batch_size:
            return
        self.updates += 1
        states, new_states, actions, rewards, dones = self.memory.sample_memory(sample_length=self.batch_size)

        next_actions, next_log_probs = self.actor.sample_normal(observation=new_states, reparameterise=True)
        target_q1 = self.critic_target1.forward(observation=new_states, action=next_actions)
        target_q2 = self.critic_target2.forward(observation=new_states, action=next_actions)
        next_min_q = torch.min(target_q1, target_q2) - self.alpha.to(self.device) * next_log_probs
        next_q = rewards + self.discount_rate * (1 - dones) * next_min_q

        curr_q1 = self.critic1.forward(observation=states, action=actions)
        curr_q2 = self.critic2.forward(observation=states, action=actions)
        q1_loss = funcs.mse_loss(curr_q1, next_q.detach())
        q2_loss = funcs.mse_loss(curr_q2, next_q.detach())

        self.critic1.optimizer.zero_grad()
        q1_loss.backward()
        self.critic1.optimizer.step()

        self.critic2.optimizer.zero_grad()
        q2_loss.backward()
        self.critic2.optimizer.step()
        if verbose:
            print("critic1 loss:", q1_loss, "critic2 loss:", q2_loss)

        curr_actions, curr_log_probs = self.actor.sample_normal(observation=states, reparameterise=True)
        if self.updates % self.policy_train_delay_modulus == 0:
            curr_q1_policy = self.critic1.forward(observation=states, action=curr_actions)
            curr_q2_policy = self.critic2.forward(observation=states, action=curr_actions)
            curr_min_q_policy = torch.min(curr_q1_policy, curr_q2_policy)
            policy_loss = torch.mean(self.alpha * curr_log_probs - curr_min_q_policy)

            self.actor.optimizer.zero_grad()
            policy_loss.backward()
            self.actor.optimizer.step()
            if verbose:
                print("actor loss:", policy_loss)

            self.polyak_soft_update(target=self.critic_target1, source=self.critic1)
            self.polyak_soft_update(target=self.critic_target2, source=self.critic2)

    def initial_target_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def polyak_soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * self.polyak + param.data * (1 - self.polyak))

    def play_games(self, num_games, verbose=False, display_gameplay=False):
        for _ in range(num_games):
            self._play_game(verbose, display_gameplay)

    def _play_game(self, verbose, display_gameplay):
        self.env.reset_env()
        total_reward = 0
        while not self.env.done:
            current_obs = torch.tensor(self.env.observation).reshape((1, -1)).to(self.device).type(torch.float)
            current_action = self.choose_action(current_obs)
            if display_gameplay:
                self.env.env.render()

            self.env.take_action(current_action.squeeze().cpu().numpy())
            total_reward += self.env.reward
            next_obs = torch.tensor(self.env.observation).reshape((1, -1)).to(self.device).type(torch.float)
            done = self.env.done

            self.train_models(verbose=False)

            if done:
                print("total_reward:", total_reward)
                self.episode_counter += 1
                if self.env.won:
                    self.winstreak += 1
                    self.total_wins += 1
                else:
                    self.winstreak = 0
                if verbose and self.episode_counter % 1 == 0:
                    print('{} frames in game {}, on a winstreak of {}. Total wins {}'.format(self.env.frames,
                                                                                             self.episode_counter,
                                                                                             self.winstreak,
                                                                                             self.total_wins))
            # Now store the experience for later use in training
            reward = torch.tensor([[self.env.reward]], dtype=torch.float).to(self.device)
            done = torch.tensor([[done]], dtype=torch.int).to(self.device)
            self.memory.append("observation", current_obs)
            self.memory.append("next_observation", next_obs)
            self.memory.append("action", current_action)
            self.memory.append("reward", reward)
            self.memory.append("done", done)
