import multiprocess as mp
from tqdm import tqdm
from methods import Agent, GreedyAgent, MSAAgent
from envs import DynamicQVRPEnv

import pickle


def run_agent(
    agentClass,
    env_configs : dict,
    agent_configs : dict = {},
    save_results = False,
    ):
    
    env = DynamicQVRPEnv(**env_configs)
    agent = agentClass(env, **agent_configs)
    
    rs, actions, infos = agent.run(10)
    
    res = {
        "rs" : rs,
        "actions" : actions,
        "infos" : infos
    }
    if save_results:
        pickle.dump(res, f'{agentClass.__name__}.pkl')
    return rs, actions, infos
    

def experiment(
    ):
    
    env_configs = {
        "K" : 50,
        "Q" : 100, 
        "DoD" : 0.5,
        "vehicle_capacity" : 25,
        "re_optimization" : False,
        "costs_KM" : [1, 1],
        "emissions_KM" : [.1, .3]
    }
    agents = [
        GreedyAgent,
        MSAAgent
    ]
    
    agent_configs = [
        {},
        {'n_sample' : 13,},
    ]
    
    for i in range(len(agents)):
        run_agent(
            agents[i],
            env_configs,
            agent_configs[i]
        )