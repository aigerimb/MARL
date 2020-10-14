import torch
import os
from neuralnets import Actor, Critic


class MADDPG_train:
    def __init__(self, args, idx):  
        self.args = args
        self.idx = idx 
        self.train_step = 0

        # create the network
        self.actor_network = Actor(args, idx)
        self.critic_network = Critic(args)

        # build up the target network
        self.actor_target_network = Actor(args, idx)
        self.critic_target_network = Critic(args)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args['lr_actor'])
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args['lr_critic'])

        
    # soft update
    def soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args['tau']) * target_param.data + self.args['tau'] * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args['tau']) * target_param.data + self.args['tau'] * param.data)

    # update the network
    def train(self, transitions, other_agents):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        # each agent receives only its own reward 
        r = transitions['r_%d' % self.idx]  
        obs, actions, obs_next = [], [], []  
        # create lists of info for all agents 
        for idx in range(self.args['n_agents']):
            obs.append(transitions['obs_%d' % idx])
            actions.append(transitions['action_%d' % idx])
            obs_next.append(transitions['obs_next_%d' % idx])

        # calculate the target Q value function
        action_next = []
        with torch.no_grad():
            # get next action given next obs 
            index = 0
            for idx in range(self.args['n_agents']):
                if idx== self.idx:
                    action_next.append(self.actor_target_network(obs_next[idx]))
                else:
                    # what other agents would do? other agents does not have the current agent 
                    action_next.append(other_agents[index].policy.actor_target_network(obs_next[idx]))
                    index += 1
            q_prime = self.critic_target_network(obs_next, action_next).detach()
            # the total return given q_prime(for the next action) + current reward 
            q_t = (r.unsqueeze(1) + self.args['gamma'] * q_prime).detach()

        # prediction of Q(s_t, a_t)
        q_v = self.critic_network(obs, actions)
        critic_loss = (q_t - q_v).pow(2).mean()

        # the actor loss
        # rerun selection of actions of the current agent given states to compute gradients 
        # for the rest of agents actions stay the same 
        actions[self.idx] = self.actor_network(obs[self.idx])
        actor_loss = - self.critic_network(obs, actions).mean()
        # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.soft_update_target_network()
        self.train_step += 1

    