from options import ParseParams
from utilities import make_env
from agent import Runner, Agent, test 
import numpy as np
import random
import torch
from MADDPG import MADDPG_train 
from utilities import MemoryBuffer 
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # get the params
    args = ParseParams()
    env, args = make_env(args)
  #  runner = Runner(env, args)
    if args['Train']:
        print("Training started")
         # initialize each agent 
        n_r = args['n_r']
        epsilon = args['epsilon']
        agents = []
        for a in range(args['n_agents']):
            agents.append(Agent(args, a))
        MB = MemoryBuffer(args)
    #    runner.train()
        returns = []
        for time_step in range(args['max_time_steps']):
            print("time_step: ", time_step)
            # reset the environment if reach the max epsiode length 
            if time_step % args['eps_len'] == 0:
                s = env.reset()
            all_actions = []
            actions = []
            # no need to take gradient: determinisitc policy + learning from buffer
            with torch.no_grad():
                for idx, agent in enumerate(agents):
                    a = agent.sample_action(s[idx], n_r, epsilon)
                    actions.append(a)
                    all_actions.append(a)
                    
            # for non-trainable players just assign random actions 
            for enemies in range(args['n_agents'], args['n_players']):
                all_actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            
            s_prime, r, done, info = env.step(all_actions)
            # store transitions one at a time 
            MB.add(s[:args['n_agents']], actions, s_prime[:args['n_agents']], r[:args['n_agents']])
            s = s_prime
            # if there is enough data start training with batch_size 
            if MB.length >= args['batch_size']:
                transitions = MB.sample(args['batch_size'])
                for agent in agents:
                    other_agents = agents.copy()
                    other_agents.remove(agent)
                    agent.learn(transitions, other_agents)
                    
            # store training progress 
            if time_step > 0 and time_step % args['test_rate'] == 0:
                returns.append(test(args, env, agents))
                plt.figure()
                plt.plot(range(len(returns)), returns)
                plt.xlabel('time_steps')
                plt.ylabel('average returns')
                plt.savefig('plt.pdf', format='png')
                np.savetxt("test_results.txt", returns)
            # update epsilon and noise rate with a very small increments at each time step
            n_r = max(0.05, n_r - 0.0000005)
            epsilon = max(0.05, epsilon - 0.0000005)
    else:
        print("Testing started")
     #   runner.test(args, env, agents)

