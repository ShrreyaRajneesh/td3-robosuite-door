import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, actor_learning_rate, critic_learning_rate, input_dims, tau, env, gamma=0.99, update_actor_interval=2, warmup=500, n_actions=2, max_size=1000000, layer1_size=256, layer2_size=128, batch_size=100,noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.max_action=env.action_space.high
        self.min_action=-env.action_space.low
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr=0
        self.time_step=0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        # create the networks

        self.actor = ActorNetwork(input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, n_actions=n_actions, name='actor', learning_rate=actor_learning_rate)
        self.critic_1 = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, n_actions=n_actions, name='critic_1', learning_rate=critic_learning_rate)
        self.critic_2= CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size, n_actions=n_actions, name='critic_2', learning_rate=critic_learning_rate)

       #create the target network
        self.target_actor = ActorNetwork(input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size,
                                  n_actions=n_actions, name='target_actor', learning_rate=actor_learning_rate)
        self.target_critic_1 = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size,
                                      n_actions=n_actions, name='target_critic_1', learning_rate=critic_learning_rate)
        self.target_critic_2 = CriticNetwork(input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size,
                                      n_actions=n_actions, name='target_critic_2', learning_rate=critic_learning_rate)

        self.noise = noise
        self.update_network_parameters(tau=1)


    #for exploration, td3 adds noise, therefore there are multiple ways for exploration
    def choose_action(self, observation, validation=False):
        if self.time_step < self.warmup and validation is False:
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,))).to(self.actor.device)
            #to give the agent a little bit experience before it starts to take actions
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
        #adding noise
        #mu_prime = mu + T.tensor(np.random.normal(scale=self.noise), dtype=T.float).to(self.actor.device)
        #clamping because noise is added
        #mu_prime = T.clamp(mu_prime, self.min_action, self.max_action)
        noise = np.random.normal(0, self.noise, size=self.n_actions)
        mu_prime = mu + T.tensor(noise, dtype=T.float, device=self.actor.device)
        mu_prime = T.clamp(mu_prime, T.tensor(self.min_action, device=self.actor.device),
                           T.tensor(self.max_action, device=self.actor.device))

        self.time_step += 1
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    #checking if we have enough data for learning
    def learn(self):
        if self.memory.mem_ctr < self.batch_size * 10:
            return

        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)

        rewards = T.tensor(rewards, dtype=T.float).to(self.critic_1.device)
        dones = T.tensor(dones).to(self.critic_1.device)
        next_states = T.tensor(next_states, dtype=T.float).to(self.critic_1.device)
        states = T.tensor(states, dtype=T.float).to(self.critic_1.device)
        actions = T.tensor(actions, dtype=T.float).to(self.critic_1.device)
        #targets help stabilize the training process
        # --- TD3 target policy smoothing ---
        noise = T.randn((self.batch_size, self.n_actions), device=self.actor.device) * 0.2
        noise = noise.clamp(-0.5, 0.5)

        # target actor output is float32, so ensure noise is float32
        noise = noise.float()

        target_actions = self.target_actor(next_states) + noise

        # clamp according to env action limits
        target_actions = target_actions.clamp(
            T.tensor(self.min_action, device=self.actor.device, dtype=T.float32),
            T.tensor(self.max_action, device=self.actor.device, dtype=T.float32)
        )

        #noise = T.tensor(np.random.normal(scale=0.2, size=target_actions.shape)).to(self.actor.device)
       # noise = T.clamp(noise, -0.5, 0.5)
       # target_actions = target_actions + noise

        #target_actions = target_actions  + T.clamp(T.tensor(np.random.normal(scale=0.2)),-0.5,max=0.5)
        #target_actions = T.clamp(target_actions, self.min_action[0], self.max_action[0])

        next_q1 = self.target_critic_1.forward(next_states, target_actions)
        next_q2 = self.target_critic_2.forward(next_states, target_actions)

        q1 = self.critic_1.forward(states, actions)
        q2 = self.critic_2.forward(states, actions)

        next_q1[dones] = 0.0
        next_q2[dones] = 0.0
        next_q1 = next_q1.view(-1)
        next_q2 = next_q2.view(-1)
        #whichever network has the lower value of the state that is what we are choosing
        next_critic_value = T.min(next_q1, next_q2)
        #the next reward and whatever it thinks the value of next state is times gamma (discount factor)

        target = rewards + self.gamma * next_critic_value
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)

        critic_loss = q1_loss + q2_loss
        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(states, self.actor.forward(states))
        actor_loss = -T.mean(actor_q1_loss) #by default doing gradient descent but we want gradient ascent to maximize the value
        actor_loss.backward()

        self.actor.optimizer.step()
        self.update_network_parameters()


    def update_network_parameters(self, tau=None):
        if tau is None:
            tau= self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        actor_state_dict = dict(actor_params)
        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)

        for name in critic_1_state_dict:

            critic_1_state_dict[name] = tau * critic_1_state_dict[name].clone() + (1 - tau) * target_critic_1_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau * critic_2_state_dict[name].clone() + (1 - tau) * target_critic_2_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1 - tau) * target_actor_state_dict[name].clone()

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)


    def save_models(self):
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()


    def load_models(self):
        try:
            self.actor.load_checkpoint()
            self.target_actor.load_checkpoint()
            self.critic_1.load_checkpoint()
            self.critic_2.load_checkpoint()
            self.target_critic_1.load_checkpoint()
            self.target_critic_2.load_checkpoint()
            print("Successfully loaded model")
        except:
            print("Failed to load model, starting from scratch")

