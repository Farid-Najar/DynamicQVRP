from tqdm import tqdm
import numpy as np
from methods import GreedyAgent, MSAAgent, Agent, OfflineAgent, SLAgent, RLAgent
from envs import DynamicQVRPEnv

import os

import pickle

def train_agents(
    agent_configs = {}
):
    
    # We setup different scenarios
    envs_configs = [
        {
            "K" : 50,
            "Q" : 100, 
            "DoD" : 1.,
            "vehicle_capacity" : 20,
            "re_optimization" : False,
            "costs_KM" : [1],
            "emissions_KM" : [.3],
            "n_scenarios" : 500 ,
            # "test"  : True
        },
        {
            "K" : 50,
            "Q" : 100, 
            "DoD" : .9,
            "vehicle_capacity" : 20,
            "re_optimization" : False,
            "costs_KM" : [1],
            "emissions_KM" : [.3],
            "n_scenarios" : 500 ,
            # "test"  : True
        },
        {
            "K" : 50,
            "Q" : 100, 
            "DoD" : 1.,
            "vehicle_capacity" : 20,
            "re_optimization" : False,
            "costs_KM" : [1, 1],
            "emissions_KM" : [.1, .3],
            "n_scenarios" : 500 ,
            # "test"  : True
        },
        {
            "K" : 50,
            "Q" : 100, 
            "DoD" : .9,
            "vehicle_capacity" : 20,
            "re_optimization" : False,
            "costs_KM" : [1, 1],
            "emissions_KM" : [.1, .3],
            "n_scenarios" : 500 ,
            # "test"  : True
        },
        ]
    
    envs = [DynamicQVRPEnv(**env_configs) for env_configs in envs_configs]
    # env = DynamicQVRPEnv(**env_configs)
    agent = RLAgent(envs[0], load_model=False, **agent_configs)
    agent.train(envs)
    
    

if __name__ == "__main__":
    train_agents()