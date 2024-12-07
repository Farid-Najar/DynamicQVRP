import numpy as np
from tqdm import tqdm
import multiprocess as mp
from copy import deepcopy

import torch

from methods.ML.supervised import train, NN

from gymnasium import Env



class Agent:
    def __init__(self, env : Env,
                 n_workers = 5,
                 parallelize = False,
                 *args, **kwargs):
        self.env = env
        self.n_workers = n_workers
        self.parallelize = parallelize
    
    def act(self, x, *args, **kwargs):
        return int(self.env.action_space.sample())
    
    def _parallel_run(self, n, initial_instance = 0):
        def process(env, i, q):
            # a = dests[i_id][np.where(a_GTS == 0)].astype(int)
            res = dict()
            
            episode_rewards = 0.
            actions = []
            infos = []
            o, info = env.reset(initial_instance + i)
            infos.append(info)
            while True:
                a = self.act(o, env = env)
                o, r, d, _, info = env.step(a)
                episode_rewards += r
                actions.append(a)
                infos.append(info)
                if d:
                    break
            

            res['a'] = a
            res['r'] = episode_rewards
            res['info'] = infos
            q.put((i, res))
            # print(f'DP {i} done')
            return
        
        episode_rewards = np.zeros(n)
        actions = [[] for _ in range(n)]
        infos = [[] for _ in range(n)]
        
        q = mp.Manager().Queue()
        
        pbar = tqdm(total=n)
        i = 0
        while i < n:
            ps = []
            for j in range(min(n-i-1, self.n_workers)):
                ps.append(
                    mp.Process(target = process, args = (deepcopy(self.env), i+j, q, ))
                )
                ps[-1].start()
        # ps[4*i+3].start()
            i += self.n_workers
            for p in ps:
                p.join()
            pbar.update(self.n_workers)
            
        while not q.empty():
            i, d = q.get()
            episode_rewards[i] = d["r"]
            actions[i] = d["a"]
            infos[i] = d["info"]
            # pbar.update(1)
            
            
        pbar.close()
        return episode_rewards, actions, infos
    
    def _simple_run(self, n, initial_instance = 0):
        
        episode_rewards = np.zeros(n)
        actions = [[] for _ in range(n)]
        infos = [[] for _ in range(n)]
        
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
    
    def run(self, n, initial_instance = 0):
        if self.parallelize:
            return self._parallel_run(n, initial_instance)
        else:
            return self._simple_run(n, initial_instance)
                
    def train(self, episodes):
        pass
    
    
class OfflineAgent(Agent):
    
    def run(self, n, initial_instance = 0):
        
        SA_configs = dict(
            T_init = 50_000, T_limit = 1, lamb = .99999,
        )
        
        def process(env, i, q):
            # a = dests[i_id][np.where(a_GTS == 0)].astype(int)
            
            res = dict()
            env.reset(initial_instance + i)
            assignment, _, info = env.offline_solution(**SA_configs)
            episode_rewards = np.sum(
                env.quantities[assignment.astype(bool) & env.is_O_allowed]
            )
            actions = assignment[env.j:]

            res['a'] = actions
            res['r'] = episode_rewards
            res['info'] = [info]
            q.put((i, res))
            # print(f'DP {i} done')
            return
        
        episode_rewards = np.zeros(n)
        actions = [[] for _ in range(n)]
        infos = [[] for _ in range(n)]
        
        q = mp.Manager().Queue()
        
        pbar = tqdm(total=n)
        
        i=0
        while i < n:
            ps = []
            for j in range(min(n-i, self.n_workers)):
                ps.append(
                    mp.Process(target = process, args = (deepcopy(self.env), i+j, q, ))
                )
                ps[-1].start()
        # ps[4*i+3].start()
            i += self.n_workers
            for p in ps:
                p.join()
            pbar.update(self.n_workers)
            
        while not q.empty():
            i, d = q.get()
            episode_rewards[i] = d["r"]
            actions[i] = d["a"]
            infos[i] = d["info"]
            
        pbar.close()
        return episode_rewards, actions, infos
    
class GreedyAgent(Agent):
    
    def act(self, x, *args, **kwargs):
        return 1
    
class MSAAgent(Agent):
    
    def __init__(self,
                 env, 
                 horizon = 15,
                 n_sample = 7,
                 softmax = False, # if false a majority vote is applied
                 **kwargs):
        
        super().__init__(env, **kwargs)
        self.n_sample = n_sample
        self.softmax = softmax
        self.horizon = horizon
        
    def act(self, x, env = None, *args, **kwargs):
        
        score = np.zeros(self.env.action_space.n)
        
        # TODO ** Parallelize the process
        for _ in range(self.n_sample):
            if env is None:
                assignment, *_ = self.env.sample(self.horizon)

                score[0] += float(assignment[self.env.j] == 0)
            else:
                assignment, *_ = env.sample(self.horizon)

                score[0] += float(assignment[env.j] == 0)
    
        score[1] = self.n_sample - score[0]
        
        if self.softmax:
            exp_score = np.exp(score)
            return np.random.choice(
                2,
                p=exp_score/exp_score.sum(),
            )
            
        return int(np.argmax(score))
    
    
class SLAgent(Agent):
    """The supervised learning agent
    This agent is trained with the expert advice of the offline agent (VA_SA).
    """
    def __init__(
        self,
        env,
        hidden_layers = [512, 512, 256],#[1024, 1024, 512, 256] [512, 512, 256]
        n_actions: int = 1,
        load_model = True,
        # *args, 
        **kwargs
        ):
        super().__init__(env, **kwargs)
        self.model = NN(
            env.observation_space.shape[0],
            hidden_layers,
            n_actions
        )
        if load_model:
            self.model.load_state_dict(torch.load('methods/ML/models/model_SL', weights_only=True))
            
        
    def act(self, x, *args, **kwargs):
        logit = self.model.forward(torch.Tensor(x))
        return int(logit>=.5)#int(torch.argmax(logit))
    
    def train(self, episodes = 300):
        path = 'data/SL_data/'
    
        x = np.load(path+'x_downsampled_50.npy')
        y = np.load(path+'y_downsampled_50.npy')
        # x /= np.amax(x, axis=1).reshape(-1, 1)
        assert len(x) == len(y)

        train(torch.Tensor(x), torch.Tensor(y), episodes, save=True)
        self.model.load_state_dict(torch.load('model_SL', weights_only=True))
        # return super().train(episodes)