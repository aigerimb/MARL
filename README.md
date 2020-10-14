# Multi-agent-RL-Algorithms

This repository contains implementations of multi-agent reinforcment learning algorithms:

## MADDPG 
To run the algorithm implemnted based on [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) please install  [Multi-agent Particle Environment](https://github.com/openai/multiagent-particle-envs) (MPE). During installation you may encounter some bugs related to the older version of gym used for implementing MPE. Please refer to this tutorial on how to fix [installation errors](https://medium.com/@amulyareddyk97/openais-multi-agent-particle-environments-515bea61c3ad). 

There are several scenarious to run MPE, which can be specified in options.py in by setting scenrio name. By default the code will start training using 'simple_tag.py' scenario, where we aim to train predators (red circles) to catch prays (green circles). The agents get rewarded each time they colllide with green circles (prays). 
