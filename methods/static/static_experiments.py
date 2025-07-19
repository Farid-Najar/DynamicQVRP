# direc = os.path.dirname(__file__)
# pri&
# caution: path[0] is reserved for script path (or '' in REPL)
# print(str(path)+'/ppo')

from methods.static.shortcut import multi_types
from methods.static.SA_baseline import recuit, recuit_VA
from methods.static.greedy_baseline import baseline
import multiprocess as mp
import numpy as np
import pickle

from copy import deepcopy
from time import time

from envs import StaticQVRPEnv, RemoveActionEnv


def OA_experiments(
    n_simulation = 1,
    # strategy = LRI,
    random_data = False,
    cluster_data = False,
    T = 50_000,
    T_init = 10_000,
    lamb = 0.999,
    log = False,
    retain = None,
    env_configs = dict(),
    n_threads = 7,
    ):
    
    real_data = False
    if cluster_data:
        env_configs['cluster_scenario'] = True
    elif random_data:
        env_configs['uniform_scenario'] = True
    else:
        real_data = True
        
    real = "real_" if real_data else "cluster_" if cluster_data else ""
    retain_comment = f"_retain{retain}" if retain is not None else ""
    
    env = StaticQVRPEnv(obs_mode='action', **env_configs)
    np.random.seed(1917)

    def process_DP(env, i, info, q):
        t0 = time()
        res = dict()
        
        # _, info = env.reset(i)
        excess = info['excess_emission']
        coeff = np.array(env._env.emissions_KM)
        quants = np.array([
            [
                env._env.quantities[i-1] if i else 0
                for i in env.initial_routes[m]
            ]
            for m in range(len(env.initial_routes))
            ], dtype=np.int64
        )
        a = multi_types(env._env.distance_matrix, env.initial_routes, coeff, excess, quants)
        # a = dests[i_id][np.where(a_GTS == 0)].astype(int)

        # env.reset(i)
        action = np.ones(env._env.H, dtype=np.int64)
        action[a] = 0
        _, r_opt, *_, info = env.step(action)   
        # res = GameLearning(game, T=T, strategy=strategy, log = log)
        res['time'] = time() - t0
        res['sol'] = a
        res['r'] = r_opt
        res['oq'] = info['oq']
        q.put((i, res))
        print(f'DP {i} done')
        return
        # res_dict = d
        
    def process_multiple_SA(env, i, q):
        t0 = time()
        res = dict()

        # env.reset(i)
        action_SA, *_ = recuit(deepcopy(env), T_init, 0, lamb, H=T)
        # res = recuit_multiple(game, T_init = T_init, T_limit = T_limit, lamb = lamb, log=log, H=T)
        # a = np.where(action_SA == 0)[0]
        
        _, r_SA, *_, info = env.step(action_SA)
        res['time'] = time() - t0
        res['sol'] = action_SA
        res['r'] = r_SA
        res['oq'] = info['oq']
        q.put((i, res))
        print(f'SA {i} done')
        return
        
        
    def process_greedy(env, i, q):
        t0 = time()
        res = dict()
        
        # env.reset(i)
        # r_EG = EHEG(env, obs)
        rs = baseline(deepcopy(env))
        a = rs['solution'].astype(int)
        
        # ac = np.where(a == 0)[0]
        _, r, *_ , info = env.step(a)
        
        res['time'] = time() - t0
        res['sol'] = a
        res['r'] = r
        res['oq'] = info['oq']
        q.put((i, res))
        print(f'greedy {i} done')
        return
        
    q_SA = mp.Manager().Queue()
    q_DP = mp.Manager().Queue()
    q_greedy = mp.Manager().Queue()
    
    res_DP = dict()
    res_SA = dict()
    res_greedy = dict()
    

    pool =  mp.Pool(processes=n_threads)
    for i in range(n_simulation):
        _, info = env.reset(i)
        pool.apply_async(process_DP, args=(deepcopy(env), i, info, q_DP,))
        pool.apply_async(process_multiple_SA, args=(deepcopy(env), i, q_SA,))
        pool.apply_async(process_greedy, args=(deepcopy(env), i, q_greedy,))
    pool.close()
    pool.join()

    print('all done !')
    while not q_DP.empty():
        i, d = q_DP.get()
        res_DP[i] = d
        
    while not q_SA.empty():
        i, d = q_SA.get()
        res_SA[i] = d
    
    while not q_greedy.empty():
        i, d = q_greedy.get()
        res_greedy[i] = d
    
    res = {
        'res_DP' : res_DP,
        'res_SA' : res_SA,
        # 'res_A' : res_A,
        'res_greedy' : res_greedy,
    }
    
    with open(f"results/static/{real}res_compare_DP_greedy_OASA_K{env.H}_n{n_simulation}{retain_comment}.pkl","wb") as f:
        pickle.dump(res, f)
    
def run_SA_VA(
    n_simulation = 1,
    # strategy = LRI,
    cluster_data = False,
    random_data = False,
    T = 100_000,
    T_init = 10_000,
    T_limit = 0,
    lamb = 0.9999,
    retain = None,
    env_configs = dict(),
    n_threads = 7,
    ):
    
    real_data = False
    if cluster_data:
        env_configs['cluster_scenario'] = True
    elif random_data:
        env_configs['uniform_scenario'] = True
    else:
        real_data = True
        
    real = "real_" if real_data else "cluster_" if cluster_data else ""
    retain_comment = f"_retain{retain}" if retain is not None else ""
    
    env = StaticQVRPEnv(obs_mode='game', **env_configs)

    np.random.seed(1917)

    def process(env, i, q):
        t0 = time()
        res = dict()

        action_SA, *_ = recuit_VA(deepcopy(env), T_init, T_limit, lamb, H=T)
        # res = recuit_multiple(game, T_init = T_init, T_limit = T_limit, lamb = lamb, log=log, H=T)
        # a = np.where(action_SA == 0)[0]
        
        *_, d, _, info = env.step(action_SA)
        # nrmlz = np.sum(env.quantities)*env.omission_cost
        # r_SA = float(d)*(nrmlz + info['r'])/nrmlz
        r_SA = info['r']*float(d)
        res['time'] = time() - t0
        res['sol'] = action_SA
        res['a'] = info['a']
        res['r'] = r_SA
        res['oq'] = info['oq'] if d else np.sum(env._env.quantities)
        q.put((i, res))
        print(f'SA {i} done')
        return
        
    q_SA = mp.Manager().Queue()
    
    res_SA = dict()
    # env.reset(0)
    # process(env, 0, q_SA)
    # print('test passed')
    

    pool =  mp.Pool(processes=n_threads)
    for i in range(n_simulation):
        _, info = env.reset(i)
        pool.apply_async(process, args=(deepcopy(env), i, q_SA,))
    pool.close()
    pool.join()
        
    print('all done !')
    while not q_SA.empty():
        i, d = q_SA.get()
        res_SA[i] = d
    
    res = {
        'res_SA' : res_SA,
    }
    
    # with open(f"res_compare_baseline_greedy_SA_{strategy.__name__}_Q{Q}_K{K}_n{n_simulation}_T{T}.pkl","wb") as f:
    with open(f"results/static/{real}res_SA_VA_K{env._env.H}_n{n_simulation}{retain_comment}.pkl","wb") as f:
        pickle.dump(res, f)
            
            
if __name__ == '__main__' :
    
    
    OA_experiments(
        5,
        # real_data=True, 
        env_configs = {
            "vehicle_capacity" : 25,
            "re_optimization" : False,
            "emissions_KM" : [.1, .1, .3, .3],
        },
        
    )