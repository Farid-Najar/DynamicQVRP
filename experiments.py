from tqdm import tqdm

from methods import GreedyAgent, MSAAgent, Agent, OfflineAgent
from envs import DynamicQVRPEnv

import pickle


def run_agent(
    agentClass,
    env_configs : dict,
    episodes = 10,
    agent_configs : dict = {},
    save_results = False,
    title = None,
    ):
    
    env = DynamicQVRPEnv(**env_configs)
    agent = agentClass(env, **agent_configs)
    
    rs, actions, infos = agent.run(episodes)
    
    res = {
        "rs" : rs,
        "actions" : actions,
        "infos" : infos
    }
    if save_results:
        tit = 'results/'+title+'.pkl' if title is not None else f'results/{agentClass.__name__}.pkl'
        with open(tit, 'wb') as f:
            pickle.dump(res, f)
    return rs, actions, infos
    

def experiment1(
        episodes = 200,
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
    
    agents = {
        "offline" : dict(
            agentClass = OfflineAgent,
            env_configs = env_configs,
            episodes = episodes,
            agent_configs = {"n_workers": 10},
            save_results = True,
            title = "res_offline",
        ),
        "greedy" : dict(
            agentClass = GreedyAgent,
            env_configs = env_configs,
            episodes = episodes,
            agent_configs = {},
            save_results = True,
            title = "res_greedy",
        ),
        "random" : dict(
            agentClass = Agent,
            env_configs = env_configs,
            episodes = episodes,
            agent_configs = {},
            save_results = True,
            title = "res_random",
        ),
        "MSA" : dict(
            agentClass = MSAAgent,
            env_configs = env_configs,
            episodes = episodes,
            agent_configs = dict(n_sample=21, parallelize = True),
            save_results = True,
            title = "res_MSA",
        ),
    }
    
    for agent_name in agents:
        run_agent(**agents[agent_name])
        print(agent_name, "done")