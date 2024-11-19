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
    
    def __init__(self,
                 env, 
                 horizon = 15,
                 n_sample = 7,
                 softmax = False, # if false a majority vote is applied
                 **kwargs):
        
        self.env = env
        self.n_sample = n_sample
        self.softmax = softmax
        self.horizon = horizon
        
    def act(self, x):
        
        score = np.zeros(self.env.action_space.n)
        
        # TODO ** Parallelize the process
        for _ in range(self.n_sample):
            assignment, *_ = self.env.sample(self.horizon)
            
            score[0] += float(assignment[self.env.j] == 0)
    
        score[1] = self.n_sample - score[0]
        
        if self.softmax:
            exp_score = np.exp(score)
            return np.random.choice(
                2,
                p=exp_score/exp_score.sum(),
            )
            
        return int(np.argmax(score))