from tqdm import tqdm
from copy import deepcopy

import numpy as np
from methods.agent import GreedyAgent, MSAAgent, Agent, OfflineAgent, DQNAgent
from envs import DynamicQVRPEnv

from methods.static import OA_experiments, run_SA_VA, RO_greedy_experiments, different_RO_freq

from methods.static import game_experiments, EXP3, LRI, run_RL

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
    agent = agentClass(env, env_configs = env_configs, **agent_configs)
    
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
        
    env_configs['k_med'] = 7
    # if env_configs["re_optimization"] : 
    env_configs_DQN_VA_as_OA = deepcopy(env_configs)
    env_configs_DQN_VA_as_OA["vehicle_assignment"] = True
    
    env_configs_DQN_VA = deepcopy(env_configs_DQN_VA_as_OA)
    env_configs_DQN_VA["re_optimization"] = False
        
    # RL_name = f"res_RL_DQN_{va}{RL_name_comment}"
    RL_model_comment = ''
    if "cluster_scenario" in env_configs and env_configs["cluster_scenario"]:
        RL_model_comment += 'clusters_'
    elif "uniform_scenario" in env_configs and env_configs["uniform_scenario"]:
        RL_model_comment += 'uniform_'
        
    RL_model_comment += 'VRP' if len(env_configs["emissions_KM"])>1 else 'TSP'
    RL_model_comment += str(len(env_configs["emissions_KM"])) if len(env_configs["emissions_KM"])>1 else ''
    RL_model_comment += f'Q{env_configs["Q"]}'
    try:
        RL_model_comment += '_uniforme' if env_configs['noised_p'] and va=='VA' else ''
    except:
        pass
        
    if RL_model is None:
        RL_model = f'DQN_{RL_model_comment}_VA'
    agents = {
        "fafs" : dict(
            agentClass = GreedyAgent,
            env_configs = env_configs,
            episodes = episodes,
            agent_configs = {},
            save_results = True,
            title = "res_fafs",
        ),
        "random" : dict(
            agentClass = Agent,
            env_configs = env_configs,
            episodes = episodes,
            agent_configs = {},
            save_results = True,
            title = "res_random",
        ),
        # "SL" : dict(
        #     agentClass = SLAgent,
        #     env_configs = env_configs,
        #     episodes = episodes,
        #     agent_configs = dict(),
        #     save_results = True,
        #     title = "res_SL",
        # ),
        # "RL_OA" : dict(
        #     agentClass = DQNAgent,
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
            agentClass = DQNAgent,
            env_configs = env_configs_DQN_VA,
            episodes = episodes,
            agent_configs = dict(
                algo = RL_model,
                hidden_layers = RL_hidden_layers, 
            ),
            save_results = True,
            title = "res_RL_DQN_VA",
        ),
        # "RL_VA_as_OA" : dict(
        #     agentClass = DQNAgent,
        #     env_configs = env_configs_DQN_VA_as_OA,
        #     episodes = episodes,
        #     agent_configs = dict(
        #         algo = RL_model,
        #         hidden_layers = RL_hidden_layers, 
        #     ),
        #     save_results = True,
        #     title = "res_RL_DQN_VA_as_OA",
        # ),
        
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
        "offline" : dict(
            agentClass = OfflineAgent,
            env_configs = env_configs,
            episodes = episodes,
            agent_configs = {"n_workers": 7},
            save_results = True,
            title = "res_offline",
        ),
        # "MSA" : dict(
        #     agentClass = MSAAgent,
        #     env_configs = env_configs,
        #     episodes = episodes,
        #     agent_configs = dict(
        #         horizon = env_configs["horizon"], 
        #         n_sample=101, parallelize = False, 
        #         accept_bonus = 0),
        #     save_results = True,
        #     title = "res_MSA",
        # ),
        # "MSA_softmax" : dict(
        #     agentClass = MSAAgent,
        #     env_configs = env_configs,
        #     episodes = episodes,
        #     agent_configs = dict(
        #         horizon = env_configs["horizon"],
        #         n_sample=51, parallelize = False, softmax = True),
        #     save_results = True,
        #     title = "res_MSA_softmax",
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
    
    try:
        va = 'VA' if env_configs['vehicle_assignment'] else 'OA'
    except:
        va = 'OA'
        
    env_configs['k_med'] = 17
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
            RL_model_comment += '_uniforme' if env_configs['noised_p'] and va=='VA' else ''
        except:
            pass
        
    if RL_model is None:
        RL_model = f'DQN_{RL_model_comment}_VA'
    
    for dod in DoDs:
        
        path = f'results/DoD{dod:.2f}/'
        try:
            os.mkdir(path)
        except :
            pass
        
        env_configs["DoD"] = dod
        env_configs_DQN_VA["DoD"] = dod
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
            "random" : dict(
                agentClass = Agent,
                env_configs = env_configs,
                episodes = episodes,
                agent_configs = {},
                save_results = True,
                title = "res_random",
            ),
            "offline" : dict(
                agentClass = OfflineAgent,
                env_configs = env_configs,
                episodes = episodes,
                agent_configs = {"n_workers": 7},
                save_results = True,
                title = "res_offline",
            ),
            "MSA" : dict(
                agentClass = MSAAgent,
                env_configs = env_configs,
                episodes = episodes,
                agent_configs = dict(n_sample=51, parallelize = True),
                save_results = True,
                title = "res_MSA",
            ),
            # "SL" : dict(
            #     agentClass = SLAgent,
            #     env_configs = env_configs,
            #     episodes = episodes,
            #     agent_configs = dict(),
            #     save_results = True,
            #     title = "res_SL",
            # ),
            # "RL" : dict(
            #     agentClass = RLAgent,
            #     env_configs = env_configs,
            #     episodes = episodes,
            #     agent_configs = dict(),
            #     save_results = True,
            #     title = "res_RL",
            # ),
            # "RL_VA" : dict(
            #     agentClass = DQNAgent,
            #     env_configs = env_configs_DQN_VA,
            #     episodes = episodes,
            #     agent_configs = dict(
            #         algo = RL_model,
            #         hidden_layers = RL_hidden_layers, 
            #     ),
            #     save_results = True,
            #     title = "res_RL_DQN_VA",
            # ),
            # "RL_VA_as_OA" : dict(
            #     agentClass = RLAgent,
            #     env_configs = env_configs_DQN_VA_as_OA,
            #     episodes = episodes,
            #     agent_configs = dict(
            #         algo = RL_model,
            #         hidden_layers = RL_hidden_layers, 
            #     ),
            #     save_results = True,
            #     title = "res_RL_DQN_VA_as_OA",
            # ),
        }

        for agent_name in agents:
            run_agent(**agents[agent_name], path=path)
            print(agent_name, "done")
     
def experiment_Q_impact(
      steps = 100,
      env_configs = {
            "horizon" : 100,
            "Q" : 1000, 
            "DoD" : 1.,
            "vehicle_capacity" : 25,
            "re_optimization" : False,
            "emissions_KM" : [.3, .3, .3, .3],
            "test"  : True,
        },
      comment = "",
    ):
    rewards = []
    for Q in np.arange(steps, env_configs["Q"], steps):
        env_configs["Q"] = Q
        rs, *_ = run_agent(
            agentClass = OfflineAgent,
            env_configs = env_configs,
            episodes = 100,
            agent_configs = {"n_workers": 7},
            title = f"res_offline{int(Q)}",
        )
        rewards.append(rs.copy())
        print(Q, "done")
        
    np.save(f'results/rewards_Q{comment}.npy', np.array(rewards))
    
def run_offline(
        episodes = 100,
        env_configs = {
            "horizon" : 50,
            "Q" : 100, 
            "DoD" : 0.5,
            "vehicle_capacity" : 25,
            "re_optimization" : False,
            "emissions_KM" : [.1, .3],
            "n_scenarios" : 500
        },
        comment = "",
    ):
    """Compares different methods implemented so far between them on the 
    same environment.

    Parameters
    ----------
    episodes : int, optional
        The number of episodes to run for each agent, by default 200
    """
    
    # with open(f'results/static/env_configs.pkl', 'wb') as f:
    #     pickle.dump(env_configs, f)
        
    agents = {
        "offline" : dict(
            agentClass = OfflineAgent,
            env_configs = env_configs,
            episodes = episodes,
            agent_configs = {"n_workers": 7},
            save_results = True,
            title = f"res_offline_K{env_configs['horizon']}",
        ),
    }
    
    for agent_name in agents:
        run_agent(**agents[agent_name], path = f'results/static/{comment}')
        print(agent_name, "done")


if __name__ == "__main__":
    ###############################################
    #### Main experiments
    ###############################################
    
    # Studying the impact of the quota
    # experiment_Q_impact()
    # experiment_Q_impact(
    #   env_configs = {
    #         "horizon" : 100,
    #         "Q" : 1000, 
    #         "DoD" : 1.,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #     },
    #   comment = "_heterogenous",
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
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #         # "n_scenarios" : 500,
    #         # "vehicle_assignment" : True,
    #     },
    #     # RL_model='DQN_VRP4_VA',
    #     RL_hidden_layers = [1024, 1024, 1024],
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
    #         "emissions_KM" : [.1, .3],
    #         # "n_scenarios" : 500,
    #         "cluster_scenario" : True,
    #         "test"  : True,
    #         # "vehicle_assignment" : True,
    #     },
    #     RL_hidden_layers = [1024, 1024, 1024],
    # )
    
    # VRP with 4 vehicles on uniform scenarios
    # experiment(
    #     100,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50, 
    #         "DoD" : 1.,
    #         "vehicle_capacity" : 20,
    #         "re_optimization" : True,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         # "n_scenarios" : 500,
    #         "uniform_scenario" : True,
    #         "test"  : True,
    #         # "vehicle_assignment" : True,
    #     },
    #     RL_hidden_layers = [1024, 1024, 1024],
    # )
    
    # VRP with 4 vehicles on real scenarios different quantities
    # experiment(
    #     100,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50, 
    #         "DoD" : 1.,
    #         "vehicle_capacity" : 20,
    #         "re_optimization" : True,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         # "n_scenarios" : 500,
    #         "different_quantities" : True,
    #         "test"  : True,
    #         # "vehicle_assignment" : True,
    #     },
    #     RL_hidden_layers = [1024, 1024, 1024],
    # )
    
    ###############################################
    #### Static experiments
    ###############################################
    
    ###############################################
    # Finding the best re-optimization frequency
    # different_RO_freq(
    #     freqs = [1,2,5,7,10,15,50],
    #     env_configs = {
    #         "horizon" : 50,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .3],
    #         "vehicle_assignment" : False,
    #         "change_instance" : True,
    #         "test" : True
    #     },
    #     n_simulation = 20,
    # )
    
    # "horizon" : 50,
    #         "Q" : 100,
    #         "vehicle_capacity" : 25,
    #         # "uniform_scenario" : True,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .3],
    #         "test"  : True,
    
    ##############################################################################################
    #### Real experiments
    
    # game_experiments(
    #     100,
    #     EXP3,
    #     T = 10_000,
    #     real_data=True,
    #     log = False,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "vehicle_assignment" : True,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #     },
    # )
    
    # game_experiments(
    #     100,
    #     LRI,
    #     T = 10_000,
    #     real_data=True,
    #     log = False,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "vehicle_assignment" : True,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #     },
    # )
        
    # OA_experiments(
    #     100,
    #     # real_data=True, 
    #     T = 50_000,
    #     T_init = 10_000,
    #     lamb = 0.999,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #     },
    # )
    
    # RO_greedy_experiments(
    #     100,
    #     # real_data=True, 
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #         "re_optimization_freq" : 1,
    #     },
    #     comment='_ReOptimizedEvery1'
    # )
    
    # RO_greedy_experiments(
    #     100,
    #     # real_data=True, 
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #         "re_optimization_freq" : 5,
    #     },
    #     comment='_ReOptimizedEvery5'
    # )
    
    # RO_greedy_experiments(
    #     100,
    #     # real_data=True, 
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #         "re_optimization_freq" : 10,
    #     },
    #     comment='_ReOptimizedEvery10'
    # )
        
    # run_SA_VA(
    #     100,
    #     # real_data=True, 
    #     T = 75_000,
    #     T_init = 10_000,
    #     lamb = 0.999,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_assignment" : True,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #     },
    # )
    # run_offline(
    #     100,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "DoD" : 1.,
    #         "vehicle_capacity" : 25,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #     },
    #     comment = "real_",
    # )
    
    # run_RL(
    #     steps = 500_000,
    #     cluster_data=False,
    #     random_data=False,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #         "obs_mode" : "multi",
    #         "change_instance" : False,
    #     },
    #     rewards_mode= 'aq', #'normalized_terminal', 
    # )
    
    # run_RL(
    #     steps = 500_000,
    #     cluster_data=False,
    #     random_data=False,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #         "obs_mode" : "assignment",
    #         "change_instance" : False,
    #     },
    #     rewards_mode= 'aq', #'normalized_terminal', 
    # )
    
    # run_RL(
    #     steps = 500_000,
    #     cluster_data=False,
    #     random_data=False,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #         "obs_mode" : "elimination_gain",
    #         "change_instance" : False,
    #     },
    #     rewards_mode= 'aq', #'normalized_terminal', 
    # )
    
    # run_RL(
    #     steps = 500_000,
    #     cluster_data=False,
    #     random_data=False,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #         "obs_mode" : "action",
    #         "change_instance" : False,
    #     },
    #     rewards_mode= 'aq', #'normalized_terminal', 
    # )
    
    
    
    run_RL(
        steps = 1_000_000,
        cluster_data=False,
        random_data=False,
        env_configs = {
            "horizon" : 100,
            "Q" : 50,
            "vehicle_capacity" : 25,
            "re_optimization" : False,
            "emissions_KM" : [.1, .1, .3, .3],
            "test"  : True,
            "obs_mode" : "multi",
            "re_optimization_freq" : 15,
            "change_instance" : True,
        },
        rewards_mode= 'aq', #'normalized_terminal', 
    )
    
    # run_RL(
    #     steps = 1_000_000,
    #     cluster_data=False,
    #     random_data=False,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #         "obs_mode" : "assignment",
    #         "re_optimization_freq" : 10,
    #         "change_instance" : True,
    #     },
    #     rewards_mode= 'aq', #'normalized_terminal', 
    #     action_mode = "all_nodes",
    # )
    # run_RL(
    #     steps = 1_000_000,
    #     cluster_data=False,
    #     random_data=False,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #         "obs_mode" : "a+e",
    #         "re_optimization_freq" : 10,
    #         "change_instance" : True,
    #     },
    #     rewards_mode= 'aq', #'normalized_terminal', 
    #     action_mode = "all_nodes",
    # )
    
    run_RL(
        steps = 1_000_000,
        cluster_data=False,
        random_data=False,
        env_configs = {
            "horizon" : 100,
            "Q" : 50,
            "vehicle_capacity" : 25,
            "re_optimization" : False,
            "emissions_KM" : [.1, .1, .3, .3],
            "test"  : True,
            "obs_mode" : "elimination_gain",
            "re_optimization_freq" : 10,
            "change_instance" : True,
        },
        rewards_mode= 'aq', #'normalized_terminal', 
        action_mode = "destinations",
    )
    
    ####################################################################################
    #### CLuster experiments
    
    # game_experiments(
    #     100,
    #     EXP3,
    #     T = 10_000,
    #     cluster_data=True,
    #     log = False,
    #     env_configs = {
    #         "horizon" : 50,
    #         "Q" : 100,
    #         "vehicle_capacity" : 25,
    #         # "uniform_scenario" : True,
    #         "vehicle_assignment" : True,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .3],
    #     },
    # )
    
    # game_experiments(
    #     100,
    #     LRI,
    #     T = 10_000,
    #     cluster_data=True,
    #     log = False,
    #     env_configs = {
    #         "horizon" : 50,
    #         "Q" : 100,  
    #         "vehicle_capacity" : 25,
    #         "vehicle_assignment" : True,
    #         # "uniform_scenario" : True,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .3],
    #     },
    # )
        
    # OA_experiments(
    #     100,
    #     # real_data=True, 
    #     T = 50_000,
    #     T_init = 10_000,
    #     cluster_data=True,
    #     lamb = 0.999,
    #     env_configs = {
            # "horizon" : 50,
            # "Q" : 100,
            # "vehicle_capacity" : 25,
            # "emissions_KM" : [.1, .3],
            # "uniform_scenario" : True,
    #         "re_optimization" : False,
    #         "test"  : True,
    #     },
    # )
    # RO_greedy_experiments(
    #     100,
    #     # real_data=True, 
    #     cluster_data=True,
    #     env_configs = {
    #         "horizon" : 50,
    #         "Q" : 100,
    #         "vehicle_capacity" : 25,
    #         "emissions_KM" : [.1, .3],
    #         "test"  : True,
    #         "re_optimization" : False,
    #         "re_optimization_freq" : 1,
    #     },
    #     comment='_ReOptimizedEvery1'
    # )
    
    # RO_greedy_experiments(
    #     100,
    #     # real_data=True, 
    #     cluster_data=True,
    #     env_configs = {
    #         "horizon" : 50,
    #         "Q" : 100,
    #         "vehicle_capacity" : 25,
    #         "emissions_KM" : [.1, .3],
    #         "re_optimization" : False,
    #         "test"  : True,
    #         "re_optimization_freq" : 5,
    #     },
    #     comment='_ReOptimizedEvery5'
    # )
    
    # RO_greedy_experiments(
    #     100,
    #     # real_data=True, 
    #     cluster_data=True,
    #     env_configs = {
    #         "horizon" : 50,
    #         "Q" : 100,
    #         "vehicle_capacity" : 25,
    #         "emissions_KM" : [.1, .3],
    #         "re_optimization" : False,
    #         "test"  : True,
    #         "re_optimization_freq" : 10,
    #     },
    #     comment='_ReOptimizedEvery10'
    # )
    # run_SA_VA(
    #     100,
    #     # real_data=True, 
    #     cluster_data=True,
    #     T = 75_000,
    #     T_init = 10_000,
    #     lamb = 0.999,
    #     env_configs = {
    #         "horizon" : 50,
    #         "Q" : 100,  
    #         "vehicle_assignment" : True,
    #         "vehicle_capacity" : 25,
    #         # "uniform_scenario" : True,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .3],
    #         "test"  : True,
    #     },
    # )
    # run_offline(    
    #     100,
    #     env_configs = {
    #         "horizon" : 50,
    #         "Q" : 100,
    #         "DoD" : 1.,
    #         "cluster_scenario" : True,
    #         "vehicle_capacity" : 25,
    #         "emissions_KM" : [.1, .3],
    #         "test"  : True,
    #     },
    #     comment = "cluster_",
    # )
    
    
    
    # run_RL(
    #     steps = 500_000,
    #     cluster_data=True,
    #     random_data=False,
    #     env_configs = {
    #         "horizon" : 50,
    #         "Q" : 100,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .3],
    #         "test"  : True,
    #         "obs_mode" : "multi",
    #         "change_instance" : True,
    #         "re_optimization_freq" : 10,
    #     },
    #     rewards_mode= 'aq', #'normalized_terminal', 
    # )
    
    # run_RL(
    #     steps = 1_000_000,
    #     cluster_data=True,
    #     random_data=False,
    #     env_configs = {
    #         "horizon" : 50,
    #         "Q" : 100,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .3],
    #         "test"  : True,
    #         "obs_mode" : "assignment",
    #         "change_instance" : True,
    #         "re_optimization_freq" : 10,
    #     },
    #     rewards_mode= 'aq', #'normalized_terminal', 
    #     action_mode = "all_nodes",
    # )
    
    # run_RL(
    #     steps = 500_000,
    #     cluster_data=True,
    #     random_data=False,
    #     env_configs = {
    #         "horizon" : 50,
    #         "Q" : 100,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .3],
    #         "test"  : True,
    #         "obs_mode" : "elimination_gain",
    #         "re_optimization_freq" : 10,
    #         "change_instance" : True,
    #     },
    #     rewards_mode= 'aq', #'normalized_terminal', 
    #     action_mode = "destinations",
    # )
    
    # run_RL(
    #     steps = 2_000_000,
    #     cluster_data=True,
    #     random_data=False,
    #     env_configs = {
    #         "horizon" : 50,
    #         "Q" : 100,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .3],
    #         "test"  : True,
    #         "obs_mode" : "a+e",
    #         "re_optimization_freq" : 10,
    #         "change_instance" : True,
    #     },
    #     rewards_mode= 'aq', #'normalized_terminal', 
    #     action_mode = "all_nodes",
    # )
    
    # run_RL(
    #     steps = 500_000,
    #     cluster_data=True,
    #     random_data=False,
    #     env_configs = {
    #         "horizon" : 50,
    #         "Q" : 100,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .3],
    #         "test"  : True,
    #         "obs_mode" : "a+e",
    #         "re_optimization_freq" : 10,
    #         "change_instance" : True,
    #     },
    #     rewards_mode= 'aq', #'normalized_terminal', 
    #     action_mode = "destinations",
    # )
    
    ####################################################################################
    #### Uniform experiments
    
    # game_experiments(
    #     100,
    #     EXP3,
    #     T = 10_000,
    #     # real_data=True,
    #     log = False,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         # "uniform_scenario" : True,
    #         "vehicle_assignment" : True,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #     },
    # )
    
    # game_experiments(
    #     100,
    #     LRI,
    #     T = 10_000,
    #     # real_data=True,
    #     log = False,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "vehicle_assignment" : True,
    #         # "uniform_scenario" : True,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #     },
    # )
        
    # OA_experiments(
    #     100,
    #     # real_data=True, 
    #     T = 50_000,
    #     T_init = 10_000,
        # random_data=True,
    #     lamb = 0.999,
    #     env_configs = {
            # "horizon" : 100,
            # "Q" : 50,
            # "vehicle_capacity" : 25,
            # "emissions_KM" : [.1, .1, .3, .3],
    #         # "uniform_scenario" : True,
    #         "re_optimization" : False,
    #         "test"  : True,
    #     },
    # )
    # RO_greedy_experiments(
    #     100,
    #     # real_data=True, 
    #     random_data=True,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #         "re_optimization" : False,
    #         "re_optimization_freq" : 1,
    #     },
    #     comment='_ReOptimizedEvery1'
    # )
    
    # RO_greedy_experiments(
    #     100,
    #     # real_data=True, 
    #     random_data=True,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "re_optimization" : False,
    #         "test"  : True,
    #         "re_optimization_freq" : 5,
    #     },
    #     comment='_ReOptimizedEvery5'
    # )
    
    # RO_greedy_experiments(
    #     100,
    #     # real_data=True, 
    #     random_data=True,
    #     env_configs = {
    #        "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "re_optimization" : False,
    #         "test"  : True,
    #         "re_optimization_freq" : 10,
    #     },
    #     comment='_ReOptimizedEvery10'
    # )
    # run_SA_VA(
    #     100,
    #     # real_data=True, 
    #     random_data=True,
    #     T = 75_000,
    #     T_init = 10_000,
    #     lamb = 0.999,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_assignment" : True,
    #         "vehicle_capacity" : 25,
    #         # "uniform_scenario" : True,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #     },
    # )
    # run_offline(
    #     100,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "DoD" : 1.,
    #         "uniform_scenario" : True,
    #         "vehicle_capacity" : 25,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #     },
    # )
    
    # run_RL(
    #     steps = 500_000,
    #     cluster_data=False,
    #     random_data=True,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #         "obs_mode" : "multi",
    #         "change_instance" : True,
    #         "re_optimization_freq" : 10,
    #     },
    #     rewards_mode= 'aq', #'normalized_terminal', 
    # )
    
    # run_RL(
    #     steps = 500_000,
    #     cluster_data=False,
    #     random_data=True,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #         "obs_mode" : "assignment",
    #         "change_instance" : True,
    #         "re_optimization_freq" : 10,
    #     },
    #     rewards_mode= 'aq', #'normalized_terminal', 
    #     action_mode = "all_nodes",
    # )
    
    # run_RL(
    #     steps = 500_000,
    #     cluster_data=False,
    #     random_data=True,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #         "obs_mode" : "elimination_gain",
    #         "change_instance" : True,
    #         "re_optimization_freq" : 10,
    #     },
    #     rewards_mode= 'aq', #'normalized_terminal', 
    #     action_mode = "destinations",
    # )
    
    # run_RL(
    #     steps = 500_000,
    #     cluster_data=True,
    #     random_data=False,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #         "obs_mode" : "a+e",
    #         "re_optimization_freq" : 10,
    #         "change_instance" : True,
    #     },
    #     rewards_mode= 'aq', #'normalized_terminal', 
    #     action_mode = "all_nodes",
    # )
    
    # run_RL(
    #     steps = 500_000,
    #     cluster_data=True,
    #     random_data=False,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #         "obs_mode" : "a+e",
    #         "re_optimization_freq" : 10,
    #         "change_instance" : True,
    #     },
    #     rewards_mode= 'aq', #'normalized_terminal', 
    #     action_mode = "destinations",
    # )
    
    #### Real experiments with different quantities
    
    # game_experiments(
    #     100,
    #     EXP3,
    #     T = 10_000,
    #     real_data=True,
    #     log = False,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "vehicle_assignment" : True,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "different_quantities" : True,
    #     },
    #     comment = '_different_quantities',
    # )
    
    # game_experiments(
    #     100,
    #     LRI,
    #     T = 10_000,
    #     real_data=True,
    #     log = False,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "vehicle_assignment" : True,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "different_quantities" : True,
    #     },
    #     comment = '_different_quantities',
    # )
        
    # OA_experiments(
    #     100,
    #     # real_data=True, 
    #     T = 50_000,
    #     T_init = 10_000,
    #     lamb = 0.999,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #         "different_quantities" : True,
    #     },
    #     comment = '_different_quantities',
    # )
    # run_SA_VA(
    #     100,
    #     # real_data=True, 
    #     T = 75_000,
    #     T_init = 10_000,
    #     lamb = 0.999,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_assignment" : True,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #         "different_quantities" : True,
    #     },
    #     comment = '_different_quantities',
    # )
    # run_offline(
    #     100,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "DoD" : 1.,
    #         "vehicle_capacity" : 25,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #         "different_quantities" : True,
    #     },
    #     comment = "different_quantities_",
    # )
    
    
    
    # run_RL(
    #     steps = 1_000_000,
    #     cluster_data=False,
    #     random_data=False,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #         "obs_mode" : "multi_q",
    #         "change_instance" : True,
    #         "different_quantities" : True,
    #     },
    #     rewards_mode= 'aq', #'normalized_terminal', 
    # )
    
    # run_RL(
    #     steps = 1_000_000,
    #     cluster_data=False,
    #     random_data=False,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #         "obs_mode" : "assignment_q",
    #         "change_instance" : True,
    #         "different_quantities" : True,
    #     },
    #     rewards_mode= 'aq', #'normalized_terminal', 
    #     action_mode = "destinations",
    # )
    
    # run_RL(
    #     steps = 1_000_000,
    #     cluster_data=False,
    #     random_data=False,
    #     env_configs = {
    #         "horizon" : 100,
    #         "Q" : 50,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #         "obs_mode" : "elimination_gain",
    #         "change_instance" : True,
    #         "different_quantities" : True,
    #     },
    #     rewards_mode= 'aq', #'normalized_terminal', 
    #     action_mode = "destinations",
    # )
    
    
    
    
    
    ###############################################
    #### Other experiments
    ###############################################
    
    
    
    # VRP with 2 vehicles
    # experiment(
    #     100,
    #     env_configs = {
    #         "horizon" : 50,
    #         "Q" : 70, 
    #         "DoD" : 0.7,
    #         "vehicle_capacity" : 25,
    #         "re_optimization" : True,
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
    #         "emissions_KM" : [.1, .1, .3, .3],
    #         "test"  : True,
    #         # "n_scenarios" : 500,
    #         "vehicle_assignment" : True,
    #     },
    #     RL_model='DQN_VRP4Q100_VA',
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
    #         "emissions_KM" : [.1, .3],
    #         "test"  : True,
    #         "noised_p" : True,
    #         "unknown_p" : True,
    #         # "n_scenarios" : 500,
    #         # "vehicle_assignment" : True,
    #     },
    # )
    
    # TSP
    # experiment(
    #     100,
    #     env_configs = {
    #         "horizon" : 50,
    #         "Q" : 150, 
    #         "DoD" : 0.8,
    #         "vehicle_capacity" : 30,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.3],
    #         "n_scenarios" : 500  
    #     },
    # )
    
    # TSP full dynamic
    # experiment(
    #     100,
    #     env_configs = {
    #         "horizon" : 50,
    #         "Q" : 100, 
    #         "DoD" : 1.,
    #         "vehicle_capacity" : 30,
    #         "re_optimization" : True,
    #         "emissions_KM" : [.3],
    #         "n_scenarios" : 100 ,
    #         "test"  : True,
    #         "different_quantities" : False,
    #     },
    #     RL_hidden_layers = [1024, 1024, 1024],
    # )
    
    # TSP full dynamic, equi probable
    # experiment(
    #     100,
    #     env_configs = {
    #         "horizon" : 50,
    #         "Q" : 100, 
    #         "DoD" : 1.,
    #         "vehicle_capacity" : 30,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.3],
    #         "n_scenarios" : 100 ,
    #         "test"  : True,
    #         "unknown_p" : True
    #     },
    # )
    
    # TSP different DoDs
    # experiment_DoD(
    #     100,
    #     # DoDs = [1., .95, .9, .85, .8, .75, .65, .5],#[1., .95, .9, .85, .8, .75, .7, .65, .6]
    #     DoDs = np.arange(.6, 1., .05),
    #     env_configs = {
    #         "horizon" : 50,
    #         "Q" : 100, 
    #         "DoD" : 1.,
    #         "vehicle_capacity" : 30,
    #         "re_optimization" : False,
    #         "emissions_KM" : [.3],
    #         # "n_scenarios" : 500 ,
    #         "test"  : True
    #     },
    #     RL_hidden_layers = [1024, 1024, 1024],
    # )