import numpy as np
from tqdm import tqdm
import multiprocess as mp
from copy import deepcopy

import torch

from methods.ML.supervised import train, NN
from methods.ML.RL import train_RL, train_DQN

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, DQN

from gymnasium import Env



class Agent:
    def __init__(self, env : Env,
                 n_workers = 2,
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
            T_init = 1_000, T_limit = 1, lamb = .9995,
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
        
        SA_configs = dict(
            T_init = 1_000, T_limit = 1, lamb = .995,
        )
        
        env = self.env if env is None else env
        
        def process(env, i, q):
            # a = dests[i_id][np.where(a_GTS == 0)].astype(int)
            # res = dict()
            
            assignment, *_ = env.sample(self.horizon, SA_configs = SA_configs)

            score = float(assignment[env.j] == 0)
            q.put((i, score))
            # print(f'DP {i} done')
            return
        
        q = mp.Manager().Queue()
        score = np.zeros(self.env.action_space.n)
        
        pool = mp.Pool(processes=6)
        for i in range(self.n_sample):
            pool.apply_async(process, args=(deepcopy(env), i, q, ))
        # ps[4*i+3].start()
        pool.close()
        pool.join()
        while not q.empty():
            i, s = q.get()
            score[0] += s
            # pbar.update(1)
            
            
        
        
        # # TODO ** Parallelize the process
        # for _ in range(self.n_sample):
        #     if env is None:
        #         assignment, *_ = self.env.sample(self.horizon, SA_configs = SA_configs)

        #         score[0] += float(assignment[self.env.j] == 0)
        #     else:
        #         assignment, *_ = env.sample(self.horizon, SA_configs = SA_configs)

        #         score[0] += float(assignment[env.j] == 0)
    
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
        self.model.load_state_dict(torch.load('methods/ML/models/model_SL', weights_only=True))
        # return super().train(episodes)
 
 
class RLAgent(Agent):
    """The reinforcement learning agent
    This agent is trained with trial and error.
    """
    def __init__(
        self,
        env : Env,
        hidden_layers = [512, 512, 256],#[1024, 1024, 512, 256] [512, 512, 256]
        algo = 'DQN',
        load_model = True,
        # *args, 
        **kwargs
        ):
        super().__init__(env, **kwargs)
        self.model = NN(
            env.observation_space.shape[0],
            hidden_layers,
            env.action_space.n
        )
        self.algo = algo
        if load_model:
            self.model.load_state_dict(torch.load(f'methods/ML/models/model_{algo}', weights_only=True))
            # self.model = torch.load(f'methods/ML/models/model_{algo}', weights_only=False)
            
        
    def act(self, x, *args, **kwargs):
        with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
            return self.model(torch.Tensor(x)).max(0).indices.item()
    
    def train(self, episodes = 20):
            
        train_DQN(
            self.env,
            hidden_layers = [512, 512, 256],
            EPOCHS = episodes,
            BATCH_SIZE = 128,
            GAMMA = 0.99,
            EPS_START = 0.9,
            EPS_END = 0.05,
            EPS_DECAY = 1000,
            TAU = 0.05,
            LR = 1e-4,
            save = True
        )
        self.model.load_state_dict(torch.load(f'methods/ML/models/model_{self.algo}', weights_only=True))
         
      
class RLAgentSB3(Agent):
    """The reinforcement learning agent
    This agent is trained with trial and error.
    """
    def __init__(
        self,
        env : Env,
        hidden_layers = [512, 512, 256],#[1024, 1024, 512, 256] [512, 512, 256]
        algo = 'DQN', # [DQN, PPO]
        load_model = True,
        # *args, 
        **kwargs
        ):
        super().__init__(env, **kwargs)
        # self.model = NN(
        #     env.observation_space.shape[0],
        #     hidden_layers,
        #     env.action_space.n
        # )
        self.algo = DQN if algo.upper() == 'DQN' else PPO
        
        self.name = algo
        self.hidden_layers = hidden_layers
        
        
        self.model = self.algo("MlpPolicy", env, verbose=1)#, device="mps")
        
        
        if load_model:
            # self.model.load_state_dict(torch.load(f'methods/ML/models/model_{algo}', weights_only=True))
            # self.model = self.algo.load(f'methods/ML/models/{self.algo.__name__}/best_model')
            self.model = self.algo.load(f'methods/ML/{self.algo.__name__}')#/best_model')
            # self.model = torch.load(f'methods/ML/models/model_{algo}', weights_only=False)
            
        
    def act(self, x, *args, **kwargs):
        a, _ = self.model.predict(x)
        return int(a)
    
    def train(self, envs = None, episodes = 20, *args, **kwargs):
        
        steps = episodes*self.env.K*len(self.env.all_dests)
        
        envs = envs if envs is not None else [self.env]
        # self.model.learn(total_timesteps=steps)
        self.model = train_RL(
            envs,
            algo = self.algo,
            policy_kwargs = dict(
                activation_fn=torch.nn.ReLU,
                share_features_extractor=True,
                net_arch=self.hidden_layers#dict(
                #    pi=[2048, 2048, 1024, 256, 64], 
                #    vf=[2048, 2048, 1024, 256, 64])
            ),
            budget = steps,
            *args, **kwargs
        )
        # self.model = self.algo.load(f'methods/ML/models/{self.algo.__name__}')
        # self.model.save(f'methods/ML/models/{self.name}')
            
        # train_DQN(
        #     self.env,
        #     hidden_layers = [512, 512, 256],
        #     EPOCHS = episodes,
        #     BATCH_SIZE = 128,
        #     GAMMA = 0.99,
        #     EPS_START = 0.9,
        #     EPS_END = 0.05,
        #     EPS_DECAY = 1000,
        #     TAU = 0.05,
        #     LR = 1e-4,
        #     save = True
        # )
        # self.model.load_state_dict(torch.load(f'methods/ML/models/model_{self.algo}', weights_only=True))
        # return super().train(episodes)