from utilities import MemoryBuffer 
import torch 
import numpy as np 
import random 
from MADDPG import MADDPG_train 
from matplotlib import pyplot as plt 
        
class Agent(object):
    
    def __init__(self, args, idx):
        self.args = args
        self.idx = idx 
        self.policy = MADDPG_train(args, idx)
    
    def sample_action(self, o, n_r, eps):
        
        #sample action from the actor 
        if np.random.uniform() >= eps:
            obs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            a = self.policy.actor_network(obs).squeeze(0)
            a = a.cpu().numpy()
            # add Gaussian noise with n_r noise rate 
            a += n_r * self.args['high_action'] * np.random.randn(*a.shape) 
            a = np.clip(a, -self.args['high_action'], self.args['high_action'])
        else:
            # sample uniformally in a given range 
            a = np.random.uniform(-self.args['high_action'], self.args['high_action'], self.args['action_shape'][self.idx])
            
        return a.copy()
    
    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)

def test(args, env, agents):
        returns = []
        for episode in range(args['test_episodes']):
            # reset the environment
            s = env.reset()
            rewards = 0
            for time_step in range(args['test_episode_len']):
                env.render()
                all_actions = []
                with torch.no_grad():
                    for idx, agent in enumerate(agents):
                        a = agent.sample_action(s[idx], 0, 0)
                        all_actions.append(a)
                for i in range(args['n_agents'], args['n_players']):
                    all_actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                s_prime, r, done, info = env.step(all_actions)
                rewards += r[0]
                s = s_prime
            returns.append(rewards)
            print('Returns is', rewards)
        return sum(returns) / args['test_episodes']
        
class Runner(object):
    
    def __init__(self, env, args):
        self.env = env
        self.args = args 
        self.MB = MemoryBuffer(args)
        self.n_r = args['n_r']
        self.epsilon = args['epsilon']
        # initialize each agent 
        self.agents = []
        for a in range(args['n_agents']):
            self.agents.append(Agent(args, a))
        
            
    def train(self):
        returns = []
        for time_step in range(self.args['max_time_steps']):
            print("time_step: ", time_step)
            # reset the environment if reach the max epsiode length 
            if time_step % self.args['eps_len'] == 0:
                s = self.env.reset()
            all_actions = []
            actions = []
            # no need to take gradient: determinisitc policy + learning from buffer
            with torch.no_grad():
                for idx, agent in enumerate(self.agents):
                    a = agent.sample_action(s[idx], self.n_r, self.epsilon)
                    actions.append(a)
                    all_actions.append(a)
                    
            # for non-trainable players just assign random actions 
            for enemies in range(self.args['n_agents'], self.args['n_players']):
                all_actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            
            s_prime, r, done, info = self.env.step(all_actions)
            # store transitions one at a time 
            self.MB.add(s[:self.args['n_agents']], actions, s_prime[:self.args['n_agents']], r[:self.args['n_agents']])
            s = s_prime
            # if there is enough data start training with batch_size 
            if self.MB.length >= self.args['batch_size']:
                transitions = self.MB.sample(self.args['batch_size'])
                for agent in self.agents:
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    agent.learn(transitions, other_agents)
                    
            # store training progress 
            if time_step > 0 and time_step % self.args['test_rate'] == 0:
                returns.append(self.test())
                plt.figure()
                plt.plot(range(len(returns)), returns)
                plt.xlabel('episode * ' + str(self.args['test_rate'] / self.args['eps_len']))
                plt.ylabel('average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')
                
            # update epsilon and noise rate with a very small increments at each time step
            self.n_r = max(0.05, self.n_r - 0.0000005)
            self.epsilon = max(0.05, self.epsilon - 0.0000005)
           

    def test(self):
        returns = []
        for episode in range(self.args['test_episodes']):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for time_step in range(self.args['test_episode_len']):
                self.env.render()
                all_actions = []
                with torch.no_grad():
                    for idx, agent in enumerate(self.agents):
                        a = agent.sample_action(s[idx], 0, 0)
                        all_actions.append(a)
                for i in range(self.args['n_agents'], self.args['n_players']):
                    all_actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                s_prime, r, done, info = self.env.step(all_actions)
                rewards += r[0]
                s = s_prime
            returns.append(rewards)
            print('Returns is', rewards)
        return sum(returns) / self.args['test_episodes']