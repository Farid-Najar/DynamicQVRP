# direc = os.path.dirname(__file__)
# pri&
# caution: path[0] is reserved for script path (or '' in REPL)
# print(str(path)+'/ppo')

from methods.static.shortcut import multi_types
from methods.static.SA_baseline import recuit, recuit_VA
from methods.static.greedy_baseline import baseline
from methods.static.aco import ACO_TOP
from methods.static.gurobi import top_gurobi
import multiprocess as mp
import numpy as np
import pickle

from sb3_contrib.ppo_mask import MaskablePPO


from copy import deepcopy
from time import time

from envs import StaticQVRPEnv, RemoveActionEnv


def RO_greedy_experiments(
    n_simulation = 1,
    # strategy = LRI,
    random_data = False,
    cluster_data = False,
    log = False,
    env_configs = dict(),
    n_threads = 7,
    comment = '',
    save = True,
    ):
    
    real_data = False
    if cluster_data:
        env_configs['cluster_scenario'] = True
    elif random_data:
        env_configs['uniform_scenario'] = True
    else:
        real_data = True
        
    real = "real_" if real_data else "cluster_" if cluster_data else ""
    
    env = StaticQVRPEnv(obs_mode='action', **env_configs)
    np.random.seed(42)
        
        
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
        
    q_greedy = mp.Manager().Queue()
    
    res_greedy = dict()
    

    pool =  mp.Pool(processes=n_threads)
    for i in range(n_simulation):
        _, info = env.reset(i)
        pool.apply_async(process_greedy, args=(deepcopy(env), i, q_greedy,))
    pool.close()
    pool.join()

    print('all done !')
    
    while not q_greedy.empty():
        i, d = q_greedy.get()
        res_greedy[i] = d
    
    # res = {
    #     'res_greedy' : res_greedy,
    # }
    if save:
        with open(f"results/static/{real}res_greedy_K{env.H}_n{n_simulation}{comment}.pkl","wb") as f:
            pickle.dump(res_greedy, f)
    else:
        return res_greedy
        
def different_RO_freq(
    freqs = [1,2,5,7,10,15],
    env_configs = dict(),
    n_simulation = 10,
    **kwargs,
    ):
    
    oqs = np.zeros((len(freqs), n_simulation))
    for i, freq in enumerate(freqs):
        env_configs['re_optimization_freq'] = freq
        res = RO_greedy_experiments(
            n_simulation = n_simulation,
            env_configs = env_configs,
            save = False,
            **kwargs,
            )
        oqs[i] = np.array([
            res[k]['oq']
            for k in res.keys()
        ])
    
    res = {
        'freqs' : freqs,
        'oqs' : oqs, 
    }
    with open(f"results/static/different_freqs.pkl","wb") as f:
        pickle.dump(res, f)
    
    
def OA_experiments(
    n_simulation = 1,
    # strategy = LRI,
    random_data = False,
    cluster_data = False,
    T = 50_000,
    T_init = 10_000,
    lamb = 0.999,
    log = False,
    env_configs = dict(),
    n_threads = 7,
    comment = '',
    ):
    
    real_data = False
    if cluster_data:
        env_configs['cluster_scenario'] = True
    elif random_data:
        env_configs['uniform_scenario'] = True
    else:
        real_data = True
        
    real = "real_" if real_data else "cluster_" if cluster_data else ""
    
    env = StaticQVRPEnv(obs_mode='action', **env_configs)
    np.random.seed(42)

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
        # print(env.initial_routes)
        a = multi_types(env._env.distance_matrix, env.initial_routes, coeff, excess, quants)
        # print(a+1)
        # a = dests[i_id][np.where(a_GTS == 0)].astype(int)

        # env.reset(i)
        action = np.ones(env._env.H, dtype=np.int64)
        action[a] = 0
        _, r_opt, d, _, info = env.step(action) 
        # print(excess - info['excess_emission'])
        # print(info)  
        # res = GameLearning(game, T=T, strategy=strategy, log = log)
        res['time'] = time() - t0
        res['sol'] = a
        res['r'] = r_opt
        # res['oq'] = info['oq'] if d else len(action)
        res['oq'] = env._env.quantities[a].sum() if d else env._env.quantities.sum()
        # print(res['oq'])
        q.put((i, res))
        # print(f'DP {i} done')
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
    
    _, info = env.reset(0)
    process_DP(env, 0, info, q_DP)
    assert False

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
    
    with open(f"results/static/{real}res_compare_DP_greedy_OASA_K{env.H}_n{n_simulation}{comment}.pkl","wb") as f:
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
    env_configs = dict(),
    n_threads = 7,
    comment = '',
    ):
    
    real_data = False
    if cluster_data:
        env_configs['cluster_scenario'] = True
    elif random_data:
        env_configs['uniform_scenario'] = True
    else:
        real_data = True
        
    real = "real_" if real_data else "cluster_" if cluster_data else ""
    
    env = StaticQVRPEnv(obs_mode='game', **env_configs)

    np.random.seed(42)

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
    with open(f"results/static/{real}res_SA_VA_K{env._env.H}_n{n_simulation}{comment}.pkl","wb") as f:
        pickle.dump(res, f)
            

def run_RL_experiments(
    n_simulation = 1,
    # strategy = LRI,
    random_data = False,
    cluster_data = False,
    log = False,
    env_configs = dict(),
    n_threads = 7,
    log_dir = '',
    comment = '',
    action_mode = "all_nodes",
    save = True,
    ):
    
    real_data = False
    if cluster_data:
        env_configs['cluster_scenario'] = True
    elif random_data:
        env_configs['uniform_scenario'] = True
    else:
        real_data = True
        
    real = "real_" if real_data else "cluster_" if cluster_data else ""
    
    # log_dir_change_elimination = f'methods/static/ppo_mask/{real}K{env_configs['horizon']}_rewardMode(aq)_obsMode(elimination_gain)_steps(1000000)' #all nodes

    _env = StaticQVRPEnv(**env_configs)
    env = RemoveActionEnv(_env, action_mode=action_mode)
    np.random.seed(42)
        
        
    def process(env, i, q):
        t0 = time()
        res = dict()
        actions = []
        obs, info = env.reset(i)
        model = MaskablePPO.load(log_dir+'/best_model', env=env)
        d = False
        returns = 0.
        # print(info)
        while not d:  
            a = model.predict(obs, deterministic=True, action_masks=env.action_masks())[0]
            # assert info['excess_emission'] >= 0
            obs, r, d, _, info = env.step(a)
            actions.append(a)
            returns += r
        
        
        res['time'] = time() - t0
        res['sol'] = actions
        res['r'] = returns
        res['oq'] = env._env._env.quantities.sum() - returns
        q.put((i, res))
        print(f'RL {i} done')
        return
        
    q = mp.Manager().Queue()
    
    res = dict()
    
    # test
    # process(env, 0, q)

    pool =  mp.Pool(processes=n_threads)
    for i in range(n_simulation):
        pool.apply_async(process, args=(deepcopy(env), i, q,))
    pool.close()
    pool.join()

    print('all done !')
    
    while not q.empty():
        i, d = q.get()
        res[i] = d
    
    # res = {
    #     'res_greedy' : res_greedy,
    # }
    if save:
        with open(f"results/static/{real}res_RL_K{env.H}_n{n_simulation}{comment}.pkl","wb") as f:
            pickle.dump(res, f)
    else:
        return res


def run_ACO(
    n_simulation = 1,
    # strategy = LRI,
    cluster_data = False,
    random_data = False,
    num_ants=50,
    max_iter=100,
    rho=0.1,
    alpha=1.0,
    beta=2.0,
    seed=42,
    env_configs = dict(),
    n_threads = 7,
    comment = '',
    ):
    
    real_data = False
    if cluster_data:
        env_configs['cluster_scenario'] = True
    elif random_data:
        env_configs['uniform_scenario'] = True
    else:
        real_data = True
        
    real = "real_" if real_data else "cluster_" if cluster_data else ""
    
    env = StaticQVRPEnv(obs_mode='game', **env_configs)

    np.random.seed(seed)

    def process(env, i, q):
        t0 = time()
        res = dict()

        cost_matrices = env._env.cost_matrix
        n_customers = cost_matrices.shape[1] - 1

        rewards = np.zeros(n_customers + 1)  # q[0] = 0
        rewards[1:] = env._env.quantities
        Q = env._env.Q   # global total emission budget
        Cap = env._env.max_capacity    # max stops per vehicle
        aco = ACO_TOP(
            cost_matrices=cost_matrices,
            rewards=rewards,
            Q=Q,
            Cap=Cap,
            num_ants=num_ants,
            max_iter=max_iter,
            rho=rho,
            alpha=alpha,
            beta=beta,
            seed=seed,
        )

        routes, reward, total_e = aco.solve()
        # res = recuit_multiple(game, T_init = T_init, T_limit = T_limit, lamb = lamb, log=log, H=T)
        # a = np.where(action_SA == 0)[0]
        print(reward)
        # print(routes)
        # print(np.array(routes[0][1:-1])-1)
        # reward = np.array([env._env.quantities[np.array(route[1:-1])-1].sum() for route in routes]).sum()
        # print(reward)
        
        res['time'] = time() - t0
        res['routes'] = routes
        res['r'] = reward - 1
        oq = np.sum(env._env.quantities) - reward
        res['oq'] = oq if total_e <= Q + 1e-5 else np.sum(env._env.quantities)
        q.put((i, res))
        print(f'ACO {i} done')
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
        'res_ACO' : res_SA,
    }
    
    # with open(f"res_compare_baseline_greedy_SA_{strategy.__name__}_Q{Q}_K{K}_n{n_simulation}_T{T}.pkl","wb") as f:
    with open(f"results/static/{real}res_ACO_K{env._env.H}_n{n_simulation}{comment}.pkl","wb") as f:
        pickle.dump(res, f)
            

def run_gurobi(
    n_simulation = 1,
    # strategy = LRI,
    cluster_data = False,
    random_data = False,
    timeout=0.,
    seed=42,
    env_configs = dict(),
    n_threads = 6,
    comment = '',
    ):
    
    real_data = False
    if cluster_data:
        env_configs['cluster_scenario'] = True
    elif random_data:
        env_configs['uniform_scenario'] = True
    else:
        real_data = True
        
    real = "real_" if real_data else "cluster_" if cluster_data else ""
    
    env = StaticQVRPEnv(obs_mode='game', **env_configs)

    np.random.seed(seed)

    def process(env, i, q):
        # t0 = time()
        res = dict()

        cost_matrices = env._env.cost_matrix

        Q = env._env.Q   # global total emission budget
        Cap = env._env.max_capacity    # max stops per vehicle
        routes, time, total_e = top_gurobi(
            num_vehicles=cost_matrices.shape[0],
            cost_matrices=cost_matrices,
            q=env._env.quantities,
            Q=Q,
            Cap=Cap,
            timeout=timeout,
        )

        reward = np.array([env._env.quantities[np.array(route[1:-1])-1].sum() for route in routes]).sum()
        res['time'] = time#time() - t0
        res['routes'] = routes
        res['r'] = reward
        res['emissions'] = total_e
        oq = np.sum(env._env.quantities) - reward
        res['oq'] = oq if total_e <= Q + 1e-5 else np.sum(env._env.quantities)
        q.put((i, res))
        print(f'Gurobi {i} done : {res["oq"]}')
        return
        
    q_G = mp.Manager().Queue()
    
    res_G = dict()
    # env.reset(0)
    # process(env, 0, q_G)
    # print('test passed')
    

    pool =  mp.Pool(processes=n_threads)
    for i in range(n_simulation):
        env.reset(i)
        pool.apply_async(process, args=(deepcopy(env), i, q_G,))
    pool.close()
    pool.join()
        
    print('all done !')
    while not q_G.empty():
        i, d = q_G.get()
        res_G[i] = d
    
    
    # with open(f"res_compare_baseline_greedy_SA_{strategy.__name__}_Q{Q}_K{K}_n{n_simulation}_T{T}.pkl","wb") as f:
    with open(f"results/static/{real}res_Gurobi_K{env._env.H}_n{n_simulation}{comment}.pkl","wb") as f:
        pickle.dump(res_G, f)
            

        
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