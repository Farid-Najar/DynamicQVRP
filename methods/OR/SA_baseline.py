from typing import Dict, List
import numpy as np
from envs.assignment import AssignmentEnv, GameEnv

#import itertools as it
import multiprocess as mp
from numpy import random as rd
from numpy import exp
from copy import deepcopy

def OA_SA(game : AssignmentEnv, T_init, T_limit, lamb = .99, var = False, id = 0, log = False, H = 500) :
    """
    This function finds a solution for the steiner problem
        using annealing algorithm
    :param game: the assignment game
    :param T_init: the initial temperature
    :param T_limit: the lowest temperature allowed
    :return: the solution found and the evolution of the best evaluations
    """
    num_packages = game._game.num_packages
    
    best = np.ones(num_packages, dtype=int)
    solution = best.copy()
    T = T_init
    _, r, d, _, info = game.step(best)
    eval_best = -r
    eval_solution = eval_best
    m = 0
    list_best_costs = [eval_best]
    flag100 = True
    infos = []
    
    while(T>T_limit):
        # infos['T'].append(T)
        sol = rand_neighbor(solution)
        eval_sol, info = eval_annealing(sol, game)
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
        if eval_sol < eval_best :
            if info['done']:
                best = sol.copy()
            eval_best = eval_sol
            infos.append(info)
            
        if eval_sol < eval_solution :
            prob = 1
        else :
            prob = exp((eval_best - eval_sol)/T)
        rand = rd.random()
        if rand <= prob :
            solution = sol
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

def VA_SA(env : GameEnv, T_init, T_limit, lamb = .99, var = False, id = 0, log = False, H = 500) :
    """
    This function finds a solution for the steiner problem
        using annealing algorithm
    :param game: the assignment game
    :param T_init: the initial temperature
    :param T_limit: the lowest temperature allowed
    :return: the solution found and the evolution of the best evaluations
    """
    num_packages = env.K
    
    best = np.random.randint(env.num_actions, size=env.K)
    solution = best.copy()
    T = T_init
    *_, info = env.step(best)
    eval_best = -info['r']
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
        if eval_sol < eval_best :
            if d:
                best = sol.copy()
            eval_best = eval_sol
            infos.append(info)
            
        if eval_sol < eval_solution :
            prob = 1
        else :
            prob = exp((eval_best - eval_sol)/T)
        rand = rd.random()
        if rand <= prob :
            solution = sol
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

def recuit_multiple(game : AssignmentEnv, T_init, T_limit = 2, nb_researchers = 2, lamb = .99, log = False, H=500):
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
        best, list_best_costs, info = OA_SA(g, T_init = T_init, T_limit = T_limit, lamb = lamb, id = id, log=log, H=H)
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


def eval_annealing(sol, game : AssignmentEnv, malus = 500):
    """
    This evaluates the solution of the algorithm.
    :param sol: the solution which is list of booleans
    :param graph: the graph for each we search a solution
    :param terms: the list of terminal nodes
    :param malus: the coefficient that we use to penalize bad solutions
    :return: the evaluation of the solution that is an integer
    """
    _, r, d, _, info = game.step(sol)
    info['done'] = d
    # with open('log.txt', 'w+') as f:
    #     f.write(str(info))
    
    return -r, info

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


if __name__ == '__main__' :
    # NB = 5
    # games = []
    # Q = 30
    # K = 50
    # game = AssignmentEnv(Q=Q, K=K)
    # game.reset()
    #     # games.append(game)
    
    # res = recuit_multiple(game, 2000, 1, nb_researchers=NB)
    # # import pickle
    # # with open(f"res_multiple_SA_K{K}_Q{Q}.pkl","wb") as f:
    # #     pickle.dump(res, f)
    
    # bests = np.zeros(len(res))
    # import matplotlib.pyplot as plt
    # for i in res.keys():
    #     costs = res[i]['list_best_costs']
    #     bests[i] = costs[-1]
    #     plt.semilogy(costs, label=f'Searcher {i}')
    
    # sol = res[np.argmax(bests)]['sol']
    # print('solution : ', sol)
    # plt.title('Best solution costs in multiple-SA')
    # plt.legend()
    # plt.show()
    import pickle
    K = 50
    with open(f'TransportersDilemma/RL/game_K{K}.pkl', 'rb') as f:
            g = pickle.load(f)
    routes = np.load(f'TransportersDilemma/RL/routes_K{K}.npy')
    dests = np.load(f'TransportersDilemma/RL/destinations_K{K}.npy')
    env = GameEnv(AssignmentEnv(game = g, saved_routes = routes, saved_dests=dests, 
                        obs_mode='elimination_gain', 
                          change_instance = False, instance_id = 0))
    env.reset()
    VA_SA(env, 1000, 1)