from tqdm import tqdm
import numpy as np
from methods.agent import DQNAgent
from envs import DynamicQVRPEnv

import os

import pickle

def train_agents(
    agent_configs = {}
):
    
    # We setup different scenarios
    envs_configs = [
        {
            "horizon" : 50,
            "Q" : 100, 
            "DoD" : 1.,
            "vehicle_capacity" : 30,
            "re_optimization" : False,
            "costs_KM" : [1],
            "emissions_KM" : [.3],
            "n_scenarios" : 500 ,
            # "test"  : True
        },
        {
            "horizon" : 50,
            "Q" : 50, 
            "DoD" : 1,
            "vehicle_capacity" : 20,
            "re_optimization" : False,
            "costs_KM" : [1],
            "emissions_KM" : [.3],
            "n_scenarios" : 500 ,
            # "test"  : True
        },
        {
            "horizon" : 50,
            "Q" : 100, 
            "DoD" : .9,
            "vehicle_capacity" : 30,
            "re_optimization" : False,
            "costs_KM" : [1],
            "emissions_KM" : [.3],
            "n_scenarios" : 500 ,
            # "test"  : True
        },
        {
            "horizon" : 50,
            "Q" : 100, 
            "DoD" : .7,
            "vehicle_capacity" : 20,
            "re_optimization" : False,
            "costs_KM" : [1],
            "emissions_KM" : [.3],
            "n_scenarios" : 500 ,
            # "test"  : True
        },
        # {
        #     "K" : 50,
        #     "Q" : 100, 
        #     "DoD" : 1.,
        #     "vehicle_capacity" : 20,
        #     "re_optimization" : True,
        #     "costs_KM" : [1, 1],
        #     "emissions_KM" : [.1, .3],
        #     "n_scenarios" : 500 ,
        #     # "test"  : True
        # },
        # {
        #     "K" : 50,
        #     "Q" : 100, 
        #     "DoD" : .9,
        #     "vehicle_capacity" : 20,
        #     "re_optimization" : True,
        #     "costs_KM" : [1, 1],
        #     "emissions_KM" : [.1, .3],
        #     "n_scenarios" : 500 ,
        #     # "test"  : True
        # },
        ]
    
    envs = [DynamicQVRPEnv(**env_configs) for env_configs in envs_configs]
    # env = DynamicQVRPEnv(**env_configs)
    agent = DQNAgent(envs[0], load_model=False, algo='DQN',**agent_configs)
    agent.train(episodes=15)
    
    

if __name__ == "__main__":
    train_agents()