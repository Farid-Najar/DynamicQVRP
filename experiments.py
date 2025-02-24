from tqdm import tqdm
from copy import deepcopy

import numpy as np
from methods import GreedyAgent, MSAAgent, Agent, OfflineAgent, SLAgent, RLAgent
from envs import DynamicQVRPEnv

import os

import pickle


def run_agent(
    agentClass,
    env_configs : dict,
    episodes = 10,
    agent_configs : dict = {},
    save_results = False,
    title = None,
    path = 'results/'
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
        tit = path+title+'.pkl' if title is not None else f'results/{agentClass.__name__}.pkl'
        with open(tit, 'wb') as f:
            pickle.dump(res, f)
            
    del agent
    
    return rs, actions, infos
    

def experiment(
        episodes = 200,
        env_configs = {
            "horizon" : 50,
            "Q" : 100, 
            "DoD" : 0.5,
            "vehicle_capacity" : 25,
            "re_optimization" : False,
            "costs_KM" : [1, 1],
            "emissions_KM" : [.1, .3],
            "n_scenarios" : 500
        },
        RL_hidden_layers = [512, 512, 512],
        RL_model = None,
        RL_name_comment = '',
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
        
    try:
        va = 'VA' if env_configs['vehicle_assignment'] else 'OA'
    except:
        va = 'OA'
        
    # if env_configs["re_optimization"] : 
    env_configs_DQN_VA_as_OA = deepcopy(env_configs)
    env_configs_DQN_VA_as_OA["vehicle_assignment"] = True
    
    env_configs_DQN_VA = deepcopy(env_configs_DQN_VA_as_OA)
    env_configs_DQN_VA["re_optimization"] = False
        
    # RL_name = f"res_RL_DQN_{va}{RL_name_comment}"
    if "cluster_scenario" in env_configs and env_configs["cluster_scenario"]:
        RL_model_comment = 'clusters'
    else:
        RL_model_comment = 'VRP' if len(env_configs["emissions_KM"])>1 else 'TSP'
        RL_model_comment += str(len(env_configs["emissions_KM"])) if len(env_configs["emissions_KM"])>1 else ''
        try:
            RL_model_comment += f'{'_uniforme' if env_configs['noised_p'] and va=='VA' else ''}'
        except:
            pass
        
    if RL_model is None:
        RL_model = f'DQN_{RL_model_comment}_VA'
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
        # "SL" : dict(
        #     agentClass = SLAgent,
        #     env_configs = env_configs,
        #     episodes = episodes,
        #     agent_configs = dict(),
        #     save_results = True,
        #     title = "res_SL",
        # ),
        # "RL_OA" : dict(
        #     agentClass = RLAgent,
        #     env_configs = env_configs,
        #     episodes = episodes,
        #     agent_configs = dict(
        #         algo = f'DQN_{RL_model_comment}_OA',
        #         hidden_layers = RL_hidden_layers, 
        #     ),
        #     save_results = True,
        #     title = "res_RL_DQN_OA",
        # ),
        
        "RL_VA" : dict(
            agentClass = RLAgent,
            env_configs = env_configs_DQN_VA,
            episodes = episodes,
            agent_configs = dict(
                algo = RL_model,
                hidden_layers = RL_hidden_layers, 
            ),
            save_results = True,
            title = "res_RL_DQN_VA",
        ),
        "RL_VA_as_OA" : dict(
            agentClass = RLAgent,
            env_configs = env_configs_DQN_VA_as_OA,
            episodes = episodes,
            agent_configs = dict(
                algo = RL_model,
                hidden_layers = RL_hidden_layers, 
            ),
            save_results = True,
            title = "res_RL_DQN_VA_as_OA",
        ),
        
        # "RL" : dict(
        #     agentClass = RLAgent,
        #     env_configs = env_configs,
        #     episodes = episodes,
        #     agent_configs = dict(
        #         algo = 'DQN_equiProb'    
        #     ),
        #     save_results = True,
        #     title = "res_RL_DQN_equiProb",
        # ),
        # "RL" : dict(
        #     agentClass = RLAgent,
        #     env_configs = env_configs,
        #     episodes = episodes,
        #     agent_configs = dict(
        #         algo = 'PPO'    
        #     ),
        #     save_results = True,
        #     title = "res_RL_PPO",
        # ),
        # "MSA" : dict(
        #     agentClass = MSAAgent,
        #     env_configs = env_configs,
        #     episodes = episodes,
        #     agent_configs = dict(n_sample=51, parallelize = False),
        #     save_results = True,
        #     title = "res_MSA",
        # ),
        # "offline" : dict(
        #     agentClass = OfflineAgent,
        #     env_configs = env_configs,
        #     episodes = episodes,
        #     agent_configs = {"n_workers": 7},
        #     save_results = True,
        #     title = "res_offline",
        # ),
    }
    
    for agent_name in agents:
        run_agent(**agents[agent_name])
        print(agent_name, "done")
        

def experiment_DoD(
        episodes = 500,
        DoDs = [1., .95, .9, .85, .8],#, .75, .7, .65, .6],
        env_configs = {
            "horizon" : 50,
            "Q" : 100, 
            # "DoD" : 0.5,
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
    
    for dod in DoDs:
        
        path = f'results/DoD{dod}/'
        try:
            os.mkdir(path)
        except :
            pass
        
        env_configs["DoD"] = dod
        with open(f'{path}env_configs.pkl', 'wb') as f:
            pickle.dump(env_configs, f)

        agents = {
            "greedy" : dict(
                agentClass = GreedyAgent,
                env_configs = env_configs,
                episodes = episodes,
                agent_configs = {},
                save_results = True,
                title = "res_greedy",
            ),
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
            "RL" : dict(
                agentClass = RLAgent,
                env_configs = env_configs,
                episodes = episodes,
                agent_configs = dict(),
                save_results = True,
                title = "res_RL",
            ),
        }

        for agent_name in agents:
            run_agent(**agents[agent_name], path=path)
            print(agent_name, "done")
     
       
if __name__ == "__main__":
    # VRP with 2 vehicles
    # experiment(
    #     100,
    #     env_configs = {
    #         "horizon" : 50,
    #         "Q" : 70, 
    #         "DoD" : 0.7,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : True,
    #         "costs_KM" : [1, 1],
    #         "emissions_KM" : [.1, .3],
    #         "test"  : True,
    #         # "n_scenarios" : 500
    #     },
    # )
    
    # VRP full dynamic with 2 vehicles
    # experiment(
    #     100,
    #     env_configs = {
    #         "horizon" : 50,
    #         "Q" : 100, 
    #         "DoD" : 1.,
    #         "vehicle_capacity" : 20,
    #         "re_optimization" : True,
    #         "costs_KM" : [1, 1],
    #         "emissions_KM" : [.1, .3],
    #         "test"  : True,
    #         # "n_scenarios" : 500,
    #         # "vehicle_assignment" : True,
    #     },
    # )
    
    # # VRP full dynamic with 4 vehicles
    # experiment(
    #     100,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 100, 
    #         "DoD" : 1.,
    #         "vehicle_capacity" : 20,
    #         "re_optimization" : True,
    #         "costs_KM" : [1, 1, 1, 1],
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #         # "n_scenarios" : 500,
    #         "vehicle_assignment" : True,
    #     },
    #     RL_model='DQN_VRP4_VA',
    #     RL_hidden_layers = [1024, 1024, 1024],
    # )
    
    # # VRP full dynamic with 4 vehicles Q = 75
    # experiment(
    #     100,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 75, 
    #         "DoD" : 1.,
    #         "vehicle_capacity" : 20,
    #         "re_optimization" : True,
    #         "costs_KM" : [1, 1, 1, 1],
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #         # "n_scenarios" : 500,
    #         "vehicle_assignment" : True,
    #     },
    #     RL_model='DQN_VRP4Q100_VA',
    #     RL_hidden_layers = [1024, 1024, 1024],
    # )
    
    # VRP full dynamic with 4 vehicles Q = 50
    # experiment(
    #     100,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50, 
    #         "DoD" : 1.,
    #         "vehicle_capacity" : 20,
    #         "re_optimization" : True,
    #         "costs_KM" : [1, 1, 1, 1],
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #         # "n_scenarios" : 500,
    #         "vehicle_assignment" : True,
    #     },
    #     # RL_model='DQN_VRP4_VA',
    #     RL_hidden_layers = [1024, 1024, 1024],
    # )
    
    # VRP full dynamic with 2 vehicles
    # with noised probabilities
    # experiment(
    #     100,
    #     env_configs = {
    #         "horizon" : 50,
    #         "Q" : 100, 
    #         "DoD" : 1.,
    #         "vehicle_capacity" : 20,
    #         "re_optimization" : True,
    #         "costs_KM" : [1, 1],
    #         "emissions_KM" : [.1, .3],
    #         "test"  : True,
    #         "noised_p" : True,
    #         "unknown_p" : True,
    #         # "n_scenarios" : 500,
    #         # "vehicle_assignment" : True,
    #     },
    # )
    
    # VRP with 2 vehicles on cluster scenarios
    # experiment(
    #     100,
    #     env_configs = {
    #         "horizon" : 50,
    #         "Q" : 100, 
    #         "DoD" : 1.,
    #         "vehicle_capacity" : 20,
    #         "re_optimization" : True,
    #         "costs_KM" : [1, 1],
    #         "emissions_KM" : [.1, .3],
    #         # "n_scenarios" : 500,
    #         "cluster_scenario" : True,
    #         "test"  : True,
    #         # "vehicle_assignment" : True,
    #     },
    #     RL_hidden_layers = [1024, 1024, 1024],
    # )
    
    # TSP
    # experiment(
    #     500,
    #     env_configs = {
    #         "horizon" : 50,
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
        100,
        env_configs = {
            "horizon" : 50,
            "Q" : 100, 
            "DoD" : 1.,
            "vehicle_capacity" : 30,
            "re_optimization" : True,
            "costs_KM" : [1],
            "emissions_KM" : [.3],
            "n_scenarios" : 100 ,
            "test"  : True,
            "different_quantities" : False,
        },
        RL_hidden_layers = [1024, 1024, 1024],
    )
    
    # TSP full dynamic, equi probable
    # experiment(
    #     100,
    #     env_configs = {
    #         "horizon" : 50,
    #         "Q" : 100, 
    #         "DoD" : 1.,
    #         "vehicle_capacity" : 30,
    #         "re_optimization" : False,
    #         "costs_KM" : [1],
    #         "emissions_KM" : [.3],
    #         "n_scenarios" : 100 ,
    #         "test"  : True,
    #         "unknown_p" : True
    #     },
    # )
    
    # TSP different DoDs
    # experiment_DoD(
    #     500,
    #     DoDs = [.7, .65, .6],#[1., .95, .9, .85, .8, .75, .7, .65, .6]
    #     env_configs = {
    #         "horizon" : 50,
    #         "Q" : 100, 
    #         "DoD" : 1.,
    #         "vehicle_capacity" : 20,
    #         "re_optimization" : False,
    #         "costs_KM" : [1],
    #         "emissions_KM" : [.3],
    #         "n_scenarios" : 500 ,
    #         # "test"  : True
    #     },
    # )