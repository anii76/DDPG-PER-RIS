import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from algo.noise import *


# Class Actor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1):
        super(Actor, self).__init__()

        self.max_action = max_action

        hidden_1 = 128
        hidden_2 = 128
        hidden_3 = 128


        # Layer 1
        self.l1 = nn.Linear(state_dim, hidden_1)

        # Layer 2
        self.l2 = nn.Linear(hidden_1, hidden_2)

        # Layer 3
        self.l3 = nn.Linear(hidden_2, hidden_3)

        # Out layer
        self.out = nn.Linear(hidden_3, action_dim)


        self.bn1 = nn.BatchNorm1d(hidden_1)
        self.bn2 = nn.BatchNorm1d(hidden_2)
        self.bn3 = nn.BatchNorm1d(hidden_3)

    def forward(self, state):
        x = self.l1(state)
        #x = self.bn1(x)
        x = F.relu(x)

        x = self.l2(x)
        #x = self.bn2(x)
        x = F.relu(x)

        x = self.l3(x)
        #x = self.bn3(x)
        x = F.relu(x)

        action = torch.tanh(self.out(x))

        return self.max_action * action


# Class Critic
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        hidden_1 = 128
        hidden_2 = 128
        
        # Layer 1
        self.l1 = nn.Linear(state_dim + action_dim, hidden_1)


        # Layer 2
        self.l2 = nn.Linear(hidden_1, hidden_2)

        
        # Out layer
        self.out = nn.Linear(hidden_2, 1)
       

        self.bn1 = nn.BatchNorm1d(hidden_1)
        self.bn2 = nn.BatchNorm1d(hidden_2)
        

    def forward(self, state, action):
        x = self.l1(torch.cat([state, action], 1))
        #x = self.bn1(x)
        x = F.relu(x)
        
        x = self.l2(x)
        #x = self.bn2(x)
        x = F.relu(x)
        

        state_action_value = self.out(x)

        return state_action_value


# Class DDPG (Agent)
class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, device, exploration_noise, noise_scalar=1,
                 actor_lr=0.0001, critic_lr=0.001, critic_decay=1e-2, actor_decay=1 ,discount=0.99, tau=0.001, ):#noise
        self.device = device
        self.discount = discount
        self.tau = tau
        self.noise_scalar = noise_scalar
        if exploration_noise=="OU":
            self.noise = OUActionNoise(np.zeros(action_dim), sigma=noise_scalar) 
        else : 
            self.noise = AWGNActionNoise(np.zeros(action_dim), sigma=noise_scalar) 

        #Create actor and actor target
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        #Initialize actor target with the same params as actor
        self.actor_target.load_state_dict(self.actor.state_dict())
        #Initialize optimiser
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        #Create critic and critic target
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        #Initialize critic target with the same params as critic
        self.critic_target.load_state_dict(self.critic.state_dict())
        #Initialize optimiser
            #L2 weight decay spicified in DDPG paper
        self.critic_optimizer = torch.optim.Adam(self.actor.parameters(), lr=critic_lr, weight_decay=critic_decay)

    def select_action(self, state, greedy=None, test=False):
        self.actor.eval()
        if greedy == None:
            greedy = 1#self.noise_scalar

        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).to(self.device)
        noise = torch.FloatTensor(greedy * self.noise()).to(self.device) # add noise & send it to cuda device

        # In training loop
        if not test :
          action = action + noise

        return action.cpu().data.numpy().flatten()#.reshape(1, -1)

    def update_parameters(self, replay_buffer, batch_size=16):
        self.actor.train()

        # Sample from the experience replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q-value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get the current Q-value estimate
        current_Q = self.critic(state, action)

        # Compute the critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute the actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        loss_c = critic_loss.cpu().detach().numpy()
        loss_a = actor_loss.cpu().detach().numpy()
        return loss_a, loss_c


    def train(self, replay_buffer, prioritized, beta, epsilon, T, batch_size=64, IS=False): #train
        self.actor.train() #Change this with Loop T
        critic_losses = []
        actor_losses = []

        #for  _ in range(T): #nrmlm batch size
        # Sample from prioritized experience buffer
        if prioritized :
            s, a, r, s_new, done, weights, batch_idxes = replay_buffer.sample(batch_size, beta)
            #reshape data
            r = r.reshape(-1, 1)
            done = done.reshape(1, -1)

            #We do not use importance sampling weights
            #Therefore importance sampling weights are all set to 1
            #See Hyperparameter search in report
            if not IS :
                    weights = np.ones_like(r)
            weights = weights.reshape(-1, 1)

            state = torch.FloatTensor(s).to(self.device)
            action = torch.FloatTensor(a).to(self.device)
            next_state = torch.FloatTensor(s_new).to(self.device)
            not_done = torch.FloatTensor(1 - done).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

        else:
            #Uniform experience replay
            # Sample from the experience replay buffer
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
            #importance sampling weights are all set to 1
            weights, batch_idxes = np.ones_like(reward.cpu().detach().numpy()), None

        #Sqrt Weights
            #We do this since each weight will squared in MSE loss
        weights = np.sqrt(weights)
        weights = torch.FloatTensor(weights).to(self.device)

        # Compute the target Q-value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get the current Q-value estimate
        current_Q = self.critic(state, action)

        # Compute the critic loss (old loss)
        #critic_loss = F.mse_loss(current_Q, target_Q)

        # Compute the critic loss (new loss)
        TD_errors = (target_Q - current_Q)
        weighted_TD_errors = torch.mul(TD_errors, weights).to(self.device)
        # Create a zero tensor for comparaison mse
        zero_tensor = torch.zeros(weighted_TD_errors.shape).to(self.device)
        # Critic loss = MSE of weighted TD error
        critic_loss = F.mse_loss(weighted_TD_errors, zero_tensor)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute the actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # For prioritized experience replay
        # Update priorities of experiences with TD errors
        if prioritized:
            td_errors = TD_errors.detach().cpu()
            new_priorities = (
                torch.mean(torch.abs(td_errors), -1, keepdim=False)
                + epsilon
            ).squeeze(-1)
            #print(new_priorities.shape)
            #td_errors = TD_errors.detach().numpy()
            #new_priorities = np.abs(td_errors) + epsilon
            replay_buffer.update_priorities(batch_idxes, new_priorities.numpy()) #or new_priorities.numpy()?

        loss_c = critic_loss.cpu().detach().numpy()
        loss_a = actor_loss.cpu().detach().numpy()
        return loss_a, loss_c

    # Save the model parameters
    def save(self, file_name):
        torch.save(self.critic.state_dict(), file_name + "_critic")
        torch.save(self.critic_optimizer.state_dict(), file_name + "_critic_optimizer")

        torch.save(self.actor.state_dict(), file_name + "_actor")
        torch.save(self.actor_optimizer.state_dict(), file_name + "_actor_optimizer")

    # Load the model parameters
    def load(self, file_name):
        self.critic.load_state_dict(torch.load(file_name + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(file_name + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(file_name + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(file_name + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


