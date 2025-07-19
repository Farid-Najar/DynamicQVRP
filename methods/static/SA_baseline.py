from typing import Dict, List
import numpy as np

from envs import StaticQVRPEnv

#import itertools as it
import multiprocess as mp
from numpy import random as rd
from numpy import exp
from copy import deepcopy

def recuit(env : StaticQVRPEnv, T_init, T_limit, lamb = .99, id = 0, log = False, H = 500) :
    """
    This function finds a solution for the steiner problem
        using annealing algorithm
    :param env: the environment
    :param T_init: the initial temperature
    :param T_limit: the lowest temperature allowed
    :return: the solution found and the evolution of the best evaluations
    """
    num_packages = env._env.H
    
    best = np.ones(num_packages, dtype=int)
    solution = best.copy()
    T = T_init
    _, r, d, _, info = env.step(best)
    
    best_oq = info['oq'] if d else num_packages
    eval_best = -r
    eval_solution = eval_best
    m = 0
    list_best_costs = [eval_best]
    infos = []
    
    while(T>T_limit):
        # infos['T'].append(T)
        sol = rand_neighbor(solution)
        _, r, d, _, info = env.step(sol)
        oq = info['oq'] if d else num_packages
        eval_sol = -r
        # eval_sol, info = eval_annealing(sol, env)
        # infos['history'].append(info)
        
        if m%20 == 0 and log:
            print(20*'-')
            print(m)
            print('- searcher ', id)
            print('temperature : ', T)
            print('excess_emission : ', info['excess_emission'])
            print('omitted : ', info['omitted'])
            print('cost : ', eval_sol)
            print('best cost : ', eval_best)
            
        if oq <= best_oq:
            if eval_sol < eval_best:
                best = sol.copy()
                eval_best = eval_sol
                infos.append(info)
                best_oq = oq
            
        if eval_sol < eval_solution :
            prob = 1
        else :
            prob = exp((eval_best - eval_sol)/T)
        rand = rd.random()
        if rand <= prob :
            solution = sol.copy()
            eval_solution = eval_sol
        list_best_costs.append(eval_best)
        T *= lamb
        m += 1
        if m >= H:
            break
        #print(T)

    if log:
        print(f'm = {m}')
        print(eval_best)
        
    return best, list_best_costs, infos

def recuit_VA(env : StaticQVRPEnv, T_init, T_limit = 0, lamb = .99, var = False, id = 0, log = False, H = 500) :
    """
    This function finds a solution for the steiner problem
        using annealing algorithm
    :param env: the environment
    :param T_init: the initial temperature
    :param T_limit: the lowest temperature allowed
    :return: the solution found and the evolution of the best evaluations
    """
    num_packages = env._env.H
    
    best = np.random.randint(env.num_actions, size=num_packages)
    solution = best.copy()
    T = T_init
    *_, d, _, info = env.step(best)
    eval_best = -info['r']
    max_oq = np.sum(env._env.quantities)
    best_oq = info['oq'] if d else max_oq
    eval_solution = eval_best
    m = 0
    list_best_costs = [eval_best]
    flag100 = True
    infos = []
    
    while(T>T_limit):
        # infos['T'].append(T)
        sol = rand_neighbor(solution, nb_actions=env.num_actions)
        *_, d, _, info = env.step(sol)
        # info['done'] = d
        # eval_sol, info = eval_annealing(sol, game)
        eval_sol = -info['r']
        oq = info['oq'] if d else max_oq
        # infos['history'].append(info)
        
        if m%20 == 0 and log:
            print(20*'-')
            print(m)
            print('- searcher ', id)
            print('temperature : ', T)
            print('excess_emission : ', info['excess_emission'])
            print('omitted : ', info['omitted'])
            print('cost : ', eval_sol)
            print('best cost : ', eval_best)
        
        if oq <= best_oq:
            if eval_sol < eval_best :
                best = sol.copy()
                eval_best = eval_sol
                infos.append(info)
            
        if eval_sol < eval_solution :
            prob = 1
        else :
            prob = exp((eval_best - eval_sol)/T)
        rand = rd.random()
        if rand <= prob :
            solution = sol.copy()
            eval_solution = eval_sol
        list_best_costs.append(eval_best)
        T *= lamb
        m += 1
        if m >= H:
            break
        
        if(var and flag100 and T<=100):
            flag100 = False
            lamb = .999
        #print(T)

    if log:
        print(f'm = {m}')
        print(eval_best)
    
    return best, list_best_costs, infos

def recuit_multiple(game : StaticQVRPEnv, T_init, T_limit = 2, nb_researchers = 2, lamb = .99, log = False, H=500):
    """
    This function finds a solution for the steiner problem
        using annealing algorithm with multiple researchers
    :param game: the assignment game
    :param nb_researchers: the number of researchers for the best solution
    :param T_init: the initial temperature
    :param T_limit: the lowest temperature allowed
    :return: the solution found which is a set of edges
    """
    
    def process(g, id, q):
        res = dict()
        best, list_best_costs, info = recuit(g, T_init = T_init, T_limit = T_limit, lamb = lamb, id = id, log=log, H=H)
        res['sol'] = best
        res['list_best_costs'] = list_best_costs
        res['infos'] = info
        q.put((id, res))
        
    
    res = dict()
    
    ps = []
    q = mp.Manager().Queue()
    for i in range(nb_researchers):
        g = deepcopy(game)
        
        ps.append(mp.Process(target = process, args = (g, i, q)))
        ps[i].start()

    for i in range(nb_researchers):
        ps[i].join()
    
    while not q.empty():
        i, d = q.get()
        res[i] = d
        
    return res


def rand_neighbor(solution : np.ndarray, nb_changes = 1, nb_actions = 2) :
    """
    Generates new random solution.
    :param solution: the solution for which we search a neighbor
    :param nb_changes: maximum number of the changes alowed
    :return: returns a random neighbor for the solution
    """
    new_solution = solution.copy()
    i = rd.choice(len(new_solution), nb_changes, replace=False)
    if nb_actions == 2:
        new_solution[i] = 1-new_solution[i]
    else:
        candidates = list(range(nb_actions))
        candidates.remove(solution[i])
        new_solution[i] = rd.choice(candidates, nb_changes, replace=False)
    return new_solution


# if __name__ == '__main__' :
#     # NB = 5
#     # games = []
#     # Q = 30
#     # K = 50
#     # game = AssignmentEnv(Q=Q, K=K)
#     # game.reset()
#     #     # games.append(game)
    
#     # res = recuit_multiple(game, 2000, 1, nb_researchers=NB)
#     # # import pickle
#     # # with open(f"res_multiple_SA_K{K}_Q{Q}.pkl","wb") as f:
#     # #     pickle.dump(res, f)
    
#     # bests = np.zeros(len(res))
#     # import matplotlib.pyplot as plt
#     # for i in res.keys():
#     #     costs = res[i]['list_best_costs']
#     #     bests[i] = costs[-1]
#     #     plt.semilogy(costs, label=f'Searcher {i}')
    
#     # sol = res[np.argmax(bests)]['sol']
#     # print('solution : ', sol)
#     # plt.title('Best solution costs in multiple-SA')
#     # plt.legend()
#     # plt.show()
#     import pickle
#     K = 50
#     with open(f'TransportersDilemma/RL/game_K{K}.pkl', 'rb') as f:
#             g = pickle.load(f)
#     routes = np.load(f'TransportersDilemma/RL/routes_K{K}.npy')
#     dests = np.load(f'TransportersDilemma/RL/destinations_K{K}.npy')
#     env = GameEnv(AssignmentEnv(game = g, saved_routes = routes, saved_dests=dests, 
#                         obs_mode='elimination_gain', 
#                           change_instance = False, instance_id = 0))
#     env.reset()
#     recuit_tsp(env, 1000, 1)
