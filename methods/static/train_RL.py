import sys
import os
from copy import deepcopy
# direc = os.path.dirname(__file__)
from pathlib import Path
path = Path(os.path.dirname(__file__))
# pri&
# caution: path[0] is reserved for script path (or '' in REPL)
# print(str(path)+'/ppo')
sys.path.insert(1, str(path.parent.absolute()))

import argparse
import pickle

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch.nn as nn
import torch as th
from torch.nn import Linear
import torch.nn.functional as F
from gymnasium import spaces

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.evaluation import evaluate_policy as evaluate_maskable
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
# from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

# from assignment import RemoveActionEnv, NormalizedEnv
from envs import RemoveActionEnv, StaticQVRPEnv
import logging

def make_env(env, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    if env._env.change_instance:
        i = rank + seed
    else:
        i = rank
    
    def _init():
        env2 = deepcopy(env)
        env2.reset(instance_id = i, seed=seed + rank)
        return env2
    set_random_seed(seed)
    return _init

def train_RL(
    vec_env,
    algo = PPO,
    policy = "MlpPolicy",
    policy_kwargs = {},
    callbackClass=EvalCallback,
    algo_file : str|None = None,
    algo_dir = str(path)+'/ppo',
    budget = 1000,
    n_eval = 10,
    save = True,
    eval_freq = 200,
    progress_bar =True,
    n_steps = 128,
    gamma = 0.99,
):
    
    # Instantiate the agent
    if algo_file is not None:
        try:
            model = algo.load(algo_file+'/best_model', env=vec_env)
            assert model.policy_kwargs == policy_kwargs
        except Exception as e:
            logging.warning(f'couldnt load the model because this exception has been raised :\n{e}')
            
            print(f'path is {path}')
            raise('couldnt load the model!')
    else:   
        model = algo(
            policy,
            vec_env,
            policy_kwargs=policy_kwargs,
            n_steps=n_steps,
            gamma=gamma,
            batch_size=n_steps*os.cpu_count(),
            # n_epochs=50,
            # learning_rate=5e-5,
            verbose=0,
            tensorboard_log=algo_dir+"/"
        )
    logging.info(f"the model parameters :\n {model.__dict__}")
    # Train the agent and display a progress bar
    if issubclass(algo, PPO):
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=n_eval)
        logging.info(f'Before training :\n mean, std = {mean_reward}, {std_reward}')
    else:
        mean_reward, std_reward = evaluate_maskable(model, model.get_env(), n_eval_episodes=n_eval)
        logging.info(f'Before training :\n mean, std = {mean_reward}, {std_reward}')
    
    # checkpoint_callback = CheckpointCallback(
    # #   save_freq=1000,
    #   save_path="./logs/",
    #   name_prefix="rl_model",
    #   save_replay_buffer=True,
    #   save_vecnormalize=True,
    # )
    
    eval_callback = callbackClass(vec_env, best_model_save_path=algo_dir,
                             log_path=algo_dir, eval_freq=eval_freq,
                             deterministic=True, verbose=0)
    
    model.learn(
        total_timesteps=budget,
        progress_bar=progress_bar,
        log_interval=100,
        callback=eval_callback,
        # tb_log_name="ppo",
    )
    # Save the agent
    if save:
        model.save(f'{str(path)}/{algo.__name__}')
    # del model  # delete trained model to demonstrate loading
    return model

############################################################################

def train_PPO_mask(
    env_kwargs = dict(
        rewards_mode = 'terminal', # possible values ['heuristic', 'terminal']
    ),
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        share_features_extractor=True,
        net_arch=[1024, 1024, 1024, 256]#dict(
        #    pi=[2048, 2048, 1024, 256],#, 128], 
        #    vf=[2048, 2048, 1024, 256])#, 128])
    ),
    policy=MaskableActorCriticPolicy,
    n_eval = 50,
    budget = int(2e4),
    save = True,
    save_path = None,
    algo_file = str(path),
    eval_freq = 200,
    progress_bar =True,
    n_steps = 128,
    gamma = 0.99,
    instance_id = 0,
    # normalize = True,
    **kwargs
):
    if save_path is None:
        save_path = str(path)+'/ppo_mask'
    logging.basicConfig(
        filename=save_path+f'/ppo_mask.log',
        filemode='w',
        format='%(levelname)s - %(name)s :\n%(message)s \n',
        level=logging.INFO,
    )
    logging.info('Train PPO maskable started !')
    # Create environment
    # if normalize:
    #     env = NormalizedEnv(RemoveActionEnv(**env_kwargs))#rewards_mode = 'terminal')
    #     logging.info(
    #         f"""
    #         Environment information :

    #         Grid size = {env.env._env._game.grid_size}
    #         Q = {env.env._env._game.Q}
    #         K = {env.env._env._game.num_packages}
    #         n_vehicles = {env.env._env._game.num_vehicles}
    #         """
    #     )
    # else:
    env = RemoveActionEnv(**env_kwargs)
    env.reset(instance_id)
    # check_env(env)
    
    num_cpu = os.cpu_count()
    logging.info(f'Number of CPUs = {num_cpu}')
    vec_env = SubprocVecEnv([make_env(env, instance_id, seed = i) for i in range(num_cpu)])
    vec_env = VecMonitor(vec_env, save_path+"/")
    # log(type(env))
    
    model = train_RL(
        vec_env,
        algo=MaskablePPO,
        policy=policy,
        policy_kwargs=policy_kwargs,
        callbackClass=MaskableEvalCallback,
        budget=budget,
        n_eval=n_eval,
        save = save,
        algo_dir=save_path,
        algo_file = algo_file,
        eval_freq =     eval_freq ,
        progress_bar =    progress_bar,
        n_steps =     n_steps ,
        gamma = gamma,
    )

    mean_reward, std_reward = evaluate_maskable(model, model.get_env(), n_eval_episodes=10)
    logging.info(f'After training :\n mean, std = {mean_reward}, {std_reward}')
    # Create environment


    
class Multi(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, 
                 observation_space: spaces.Box, 
                 hidden_layers : list = [1024, 1024, 1024, 256],
                #  features_dim: int = 256
                 ):
        super().__init__(observation_space, hidden_layers[-1])
        # print(observation_space.sample())
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # n_input_channels = observation_space['costs'].shape[0]
        self.cnn = nn.Sequential(
            # nn.Conv2d(n_input_channels, 32, kernel_size=4, stride=4, padding=0),
            # nn.ReLU(),
            # # nn.MaxPool2d(15, 2),
            # nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            # nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()["costs"][None]).float()
            ).shape[1] + observation_space['other'].shape[0]
        
        hidden_layers.insert(0, n_flatten)
        layers = []
        for l in range(len(hidden_layers)-1):
            layers += [
                nn.Linear(hidden_layers[l], hidden_layers[l+1]),
                nn.ReLU()
        ]
        self.linear = nn.Sequential(
            *layers
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        cnn = self.cnn(observations['costs'])
        # print(cnn.shape)
        # print(observations["other"].shape)
        # print(th.cat([
        #     cnn, observations["other"]
        #     ], dim=1
        # ).shape)
        return self.linear(th.cat([
            cnn, observations["other"]
            ], dim=1
        ))
    
def run_RL(
    i = 0,
    env_configs = dict(),
    steps = 150000,
    cluster_data = False,
    random_data = False,
    rewards_mode = 'aq'
    ):
    
    real_data = False
    if cluster_data:
        env_configs['cluster_scenario'] = True
    elif random_data:
        env_configs['uniform_scenario'] = True
    else:
        real_data = True
        
    real = "real_" if real_data else "cluster_" if cluster_data else ""
    
    
    env = StaticQVRPEnv(**env_configs)
    
    if env_configs['obs_mode'] == 'multi':                
        policy = 'MultiInputPolicy'
        p_kwargs = dict(
            # normalize
            features_extractor_class=Multi,
            # features_extractor_kwargs=dict(features_dim=128),
        )
    else:
        policy = 'MlpPolicy'
        p_kwargs = dict(
            activation_fn=nn.ReLU,
            share_features_extractor=True,
            net_arch=[1024, 1024, 1024, 256]#dict(
            #    pi=[2048, 2048, 1024, 256],#, 128], 
            #    vf=[2048, 2048, 1024, 256])#, 128])
        )
    
    comment = ''
    if not env.change_instance:
        comment += f'_instanceID{str(i)}'
    train_algo = train_PPO_mask
    save_dir = str(path)+f'/ppo_mask/{real}K{env._env.H}_rewardMode({rewards_mode})_obsMode({env_configs['obs_mode']})_steps({steps})'+comment
    os.makedirs(save_dir, exist_ok=True)
    
    
    
    
    
    train_algo(
        env_kwargs = dict(
            env = env,
            rewards_mode = rewards_mode, # possible values ['heuristic', 'terminal', 'normalized_terminal', 'aq']
            action_mode = "destinations",
        ),
        instance_id = i,
        policy_kwargs = p_kwargs,
        policy=policy,
        budget=steps, n_eval=10, save = True, save_path=save_dir,
        eval_freq = 1000, progress_bar =True, n_steps = 256,
        gamma = .99, algo_file = None,
    )
        
def runs(steps = 150000, n = 50, K = 50, retain_rate = None):
    for i in range(5, n):
    # for i in range(n):
        run_RL(i, steps, K, retain_rate)
        print(f'run {i} done !')

if __name__ == '__main__':
    
    # runs(n = 10, K=50)
    # runs(n = 10, K=100)
    i = 0
    # run(i, K=100)
    print(f'run {i} done !')
    