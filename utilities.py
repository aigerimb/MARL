from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from collections import namedtuple
import random 
import numpy as np 
import threading 
# sets up the Multi-agent environemnt 
def make_env(args):
    # set scenario 
    scenario = scenarios.load(args['scenario'] + ".py").Scenario()
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    # both good and bad agents 
    args['n_players'] = env.n  
    # train only good agents
    args['n_agents'] = env.n - args['n_enemies']
    # obtain shapes of inodividual obs of agents 
    args['obs_shape'] = [env.observation_space[i].shape[0] for i in range(args['n_agents'])]  
    action_shape = []
    for content in env.action_space:
        action_shape.append(content.n)
    args['action_shape'] = action_shape[:args['n_agents']]  
    args['high_action'] = 1
    args['low_action'] = -1
    return env, args

class MemoryBuffer:
    def __init__(self, args):
        self.cap = args['buffer_size']
        self.args = args
        self.length = 0
        # create dict as memory to store info and name keys per agent
        self.memory = dict()
        for idx in range(self.args['n_agents']):
            self.memory['obs_%d' % idx] = np.empty([self.cap, self.args['obs_shape'][idx]])
            self.memory['action_%d' % idx] = np.empty([self.cap, self.args['action_shape'][idx]])
            self.memory['r_%d' % idx] = np.empty([self.cap])
            self.memory['obs_next_%d' % idx] = np.empty([self.cap, self.args['obs_shape'][idx]])
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def add(self, obs, action, obs_next, r):
        # adding one transition at a time for all agents 
        last = self.get_memory_idx(inc=1)  
        for idx in range(self.args['n_agents']):
            with self.lock:
                self.memory['obs_%d' % idx][last] = obs[idx]
                self.memory['action_%d' % idx][last] = action[idx]
                self.memory['obs_next_%d' % idx][last] = obs_next[idx]
                self.memory['r_%d' % idx][last] = r[idx]
    
    # sample the data from the replay buffer
    def sample(self, batch_size):
        transitions = {}
        idx = np.random.randint(0, self.length, batch_size)
        for key in self.memory.keys():
            transitions[key] = self.memory[key][idx]
        return transitions

    def get_memory_idx(self, inc=None):
        inc = inc or 1
        # if there is still space to store 
        if self.length + inc <= self.cap:
            last = np.arange(self.length, self.length+inc)
        # if we can store some batches in the memory and some transitions must be replaced
        elif self.length < self.cap:
            overflow = inc - (self.cap - self.length)
            idx_a = np.arange(self.length, self.cap)
            idx_b = np.random.randint(0, self.length, overflow)
            last = np.concatenate([idx_a, idx_b])
        # if we need to replace transitions 
        else:
            last = np.random.randint(0, self.cap, inc)
        self.length = min(self.cap, self.length+inc)
        if inc == 1:
            last = last[0]
        return last

