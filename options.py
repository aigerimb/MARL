import argparse


def str2bool(v):
    return v.lower() in ('true', '1')

def ParseParams():
    
    # set-uo the environemnt 
    parser = argparse.ArgumentParser(description="Multi-Agent Particles Environment")
    parser.add_argument('--scenario', default='simple_tag', help="Scenario for the World")
    parser.add_argument("--n-enemies", type=int, default=1, help="number of adversaries")
    
    # Training paprameters 
    parser.add_argument("--Train", type=bool, default=True, help="perform training")
    parser.add_argument("--buffer_size", type=int, default=10000, help="memeory buffer capacity")
    parser.add_argument("--epsilon", type=float, default=0.5, help="the value of epsilon for exploration")
    parser.add_argument("--n_r", type=float, default=0.1, help="the Gaussian noise rate")
    parser.add_argument("--lr_actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr_critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.98, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--batch_size", type=int, default=128, help="number of episodes to optimize at the same time")
    parser.add_argument("--eps_len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--max_time_steps", type=int, default=2000000, help="number of max time steps")
    
    
    # Testing parameters 
    parser.add_argument("--test_episodes", type=int, default=10, help="number of episodes for testing")
    parser.add_argument("--test_episode_len", type=int, default=100, help="length of episodes for testing")
    parser.add_argument("--test_rate", type=int, default=200, help="how often to test model")
    args, unknown = parser.parse_known_args()
    args = vars(args)
   
    
    for key, value in sorted(args.items()):
        print("{}: {}".format(key,value))
    
    return args 