from tqdm import tqdm

from methods import GreedyAgent, MSAAgent, Agent, OfflineAgent, SLAgent
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
    

def experiment(
        episodes = 200,
        env_configs = {
            "K" : 50,
            "Q" : 100, 
            "DoD" : 0.5,
            "vehicle_capacity" : 25,
            "re_optimization" : False,
            "costs_KM" : [1, 1],
            "emissions_KM" : [.1, .3],
            "n_scenarios" : 500
        },
    ):
    """Compares different methods implemented so far between them on the 
    same environment.

    Parameters
    ----------
    episodes : int, optional
        The number of episodes to run for each agent, by default 200
    """
    
    with open(f'results/env_configs.pkl', 'wb') as f:
        pickle.dump(env_configs, f)
    
    agents = {
        # "greedy" : dict(
        #     agentClass = GreedyAgent,
        #     env_configs = env_configs,
        #     episodes = episodes,
        #     agent_configs = {},
        #     save_results = True,
        #     title = "res_greedy",
        # ),
        # "random" : dict(
        #     agentClass = Agent,
        #     env_configs = env_configs,
        #     episodes = episodes,
        #     agent_configs = {},
        #     save_results = True,
        #     title = "res_random",
        # ),
        # "offline" : dict(
        #     agentClass = OfflineAgent,
        #     env_configs = env_configs,
        #     episodes = episodes,
        #     agent_configs = {"n_workers": 7},
        #     save_results = True,
        #     title = "res_offline",
        # ),
        # "MSA" : dict(
        #     agentClass = MSAAgent,
        #     env_configs = env_configs,
        #     episodes = episodes,
        #     agent_configs = dict(n_sample=21, parallelize = True),
        #     save_results = True,
        #     title = "res_MSA",
        # ),
        "SL" : dict(
            agentClass = SLAgent,
            env_configs = env_configs,
            episodes = episodes,
            agent_configs = dict(),
            save_results = True,
            title = "res_SL",
        ),
    }
    
    for agent_name in agents:
        run_agent(**agents[agent_name])
        print(agent_name, "done")
        
if __name__ == "__main__":
    # VRP with 2 vehicles
    # experiment(
    #     500,
    #     env_configs = {
    #         "K" : 50,
    #         "Q" : 70, 
    #         "DoD" : 0.7,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : True,
    #         "costs_KM" : [1, 1],
    #         "emissions_KM" : [.1, .3],
    #         "n_scenarios" : 500
    #     },
    # )
    
    # TSP
    # experiment(
    #     500,
    #     env_configs = {
    #         "K" : 50,
    #         "Q" : 150, 
    #         "DoD" : 0.8,
    #         "vehicle_capacity" : 30,
    #         "re_optimization" : False,
    #         "costs_KM" : [1],
    #         "emissions_KM" : [.3],
    #         "n_scenarios" : 500  
    #     },
    # )
    
    # TSP full dynamic
    experiment(
        500,
        env_configs = {
            "K" : 50,
            "Q" : 100, 
            "DoD" : 1.,
            "vehicle_capacity" : 30,
            "re_optimization" : False,
            "costs_KM" : [1],
            "emissions_KM" : [.3],
            "n_scenarios" : 500  
        },
    )