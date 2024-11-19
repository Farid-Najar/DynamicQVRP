import numpy as np
from tqdm import tqdm

class Agent:
    def __init__(self, env, *args, **kwargs):
        self.env = env
    
    def act(self, x):
        pass
    
    def run(self, n, initial_instance = 0):
        episode_rewards = np.zeros(n)
        actions = [[] for _ in range(n)]
        infos = [[] for _ in range(n)]
        
        # TODO ** Parallelize the process
        for i in tqdm(range(n)):
            o, info = self.env.reset(initial_instance + i)
            infos[i].append(info)
            while True:
                a = self.act(o)
                o, r, d, _, info = self.env.step(a)
                episode_rewards[i] += r
                actions[i].append(a)
                infos[i].append(info)
                if d:
                    break
                
        return episode_rewards, actions, infos
                
    def train(self, episodes):
        pass
    
    
class GreedyAgent(Agent):
    
    def act(self, x):
        return 1
    
class MSAAgent(Agent):
    #TODO *** MSA Agent : requires sampling in env
    pass