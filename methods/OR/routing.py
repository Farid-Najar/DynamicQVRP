from typing import Dict, List
import numpy as np
# from envs.assignment import AssignmentEnv, GameEnv

#import itertools as it
import multiprocess as mp
from numpy import random as rd
from numpy import exp
from copy import deepcopy
from numba import njit


@njit
def _step(
    action,
    routes,
    cost_matrix,
    distance_matrix,
    quantities,
    is_0_allowed,
    max_capacity,
    omission_cost,
    costs_KM,
    emissions_KM,
):
    info = dict()
    # routes = []#List()
    costs = np.zeros(len(cost_matrix), np.float64)
    emissions = np.zeros(len(cost_matrix), np.float64)
    # info['LCF'] = np.zeros(len(cost_matrix), np.float64)
    # info['omitted'] = alpha    
    
    if not is_0_allowed:
        action += 1
    
    a = action.copy()
    
    for i in range(len(cost_matrix)+1):
        # routes.append([])#List([0]))
        alpha = list(np.where(action == i)[0] + 1)
        if i:
            quantity = 0
            # routes[i].append(int(0))
            k = 1
            while True:
                if len(alpha) == 0:
                    break
                j = int(np.argmin(cost_matrix[i-1, routes[i-1][k-1], np.array(alpha)]))
                
                quantity += quantities[alpha[j]-1]
                
                if quantity > max_capacity:
                    # routes[0] += alpha
                    # info['LCF'][i-1] += np.sum(quantities[np.array(alpha)-1])*omission_cost
                    a[np.array(alpha) - 1] = 0
                    break
                # temp = cost_matrix[i-1, routes[i-1][-1], alpha]
                # print(cost_matrix[i-1, routes[i-1][-1], np.array(alpha)])
                # print(alpha)
                dest = alpha.pop(j)
                # if k <= max_capacity:
                costs[i-1] += distance_matrix[routes[i-1][k-1], dest]*costs_KM[i-1]
                emissions[i-1] += distance_matrix[routes[i-1][k-1], dest]*emissions_KM[i-1]
                info['LCF'][i-1] += cost_matrix[i-1, routes[i-1][k-1], dest]
                routes[i-1, k] = dest
                # print(routes[i-1], costs[i-1], emissions[i-1])
                # if k > 1:
                #     obs[routes[i-1][k-1] - 1] = cost_matrix[i-1, routes[i-1][k-2], routes[i-1][k-1]] + \
                #         cost_matrix[i-1, routes[i-1][k-1], routes[i-1][k]] - \
                #         cost_matrix[i-1, routes[i-1][k-2], routes[i-1][k]]
                
                k+=1
                
            costs[i-1] += distance_matrix[routes[i-1][k-1], 0]*costs_KM[i-1]
            emissions[i-1] += distance_matrix[routes[i-1][k-1], 0]*emissions_KM[i-1]
            info['LCF'][i-1] += cost_matrix[i-1, routes[i-1][k-1], 0]
            # routes[i].append(0)
            # if k > 1:
            #     obs[routes[i-1][k-1] - 1] = cost_matrix[i-1, routes[i-1][k-2], routes[i-1][k-1]] + \
            #             cost_matrix[i-1, routes[i-1][k-1], routes[i-1][k]] - \
            #             cost_matrix[i-1, routes[i-1][k-2], routes[i-1][k]]
                
        # else:
        #     info['omitted'] = alpha
            
    return routes, a, costs, emissions, info


def _run(env, assignment):
    routes = np.zeros((len(env.emissions_KM), env.max_capacity+2), dtype=np.int64)
    routes, a, costs, emissions, info = _step(
        assignment,
        env.routes,
        env.cost_matrix,
        env.distance_matrix,
        env.quantities,
        env.is_0_allowed,
        env.max_capacity,
        env.omission_cost,
        env.costs_KM,
        env.emissions_KM,
    )
    
    total_emission = np.sum(emissions)
    info['assignment'] = a
    info['routes'] = routes
    info['costs per vehicle'] = costs
    info['omitted'] = np.where(a==0)[0]
    info['remained_quota'] = env.Q - total_emission
    
    
def insertion(env, assignment, i):
    best = 0
    *_, info = env.step(assignment)
    eval_best = -info['r']
    best_info = deepcopy(info)
    best_routes = env.routes.copy()
    
    for v in range(1, len(env.costs_KM) + 1):
        assignment[i] = v
        *_, d, info = env.step(assignment)
        r = info['r']
        if r > eval_best and d:
            eval_best = r
            best = v
            best_info = deepcopy(info)
            best_routes = env.routes.copy()
            
    assignment[i] = best
    return assignment, best_routes, best_info
        

def SA_routing(env, action_mask : np.ndarray,
               T_init = 5000, T_limit = 1, lamb = .999, var = False, id = 0, log = False, H = 500) :
    """
    This function finds a solution for the steiner problem
        using annealing algorithm
    :param game: the assignment game
    :param T_init: the initial temperature
    :param T_limit: the lowest temperature allowed
    :return: the solution found and the evolution of the best evaluations
    """
    
    best = np.ones(env.K, int)
    best[~action_mask] = 0
    solution = best.copy()
    T = T_init
    *_, info = env.step(best)
    best_routes = env.routes.copy()
    eval_best = -info['r']
    best_info = deepcopy(info)
    eval_solution = eval_best
    m = 0
    list_best_costs = [eval_best]
    flag100 = True
    # infos = []
    
    if env.num_actions <= 2:
        return best, best_routes, best_info
    
    while(T>T_limit):
        
        # infos['T'].append(T)
        sol = rand_neighbor(solution, action_mask, nb_actions=env.num_actions)
        *_, d, _, info = env.step(sol)
        eval_sol = -info['r']
        
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
                best_routes = env.routes.copy()
                best_info = deepcopy(info)
            eval_best = eval_sol
            # infos.append(info)
            
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
    
    return best, best_routes, best_info#, list_best_costs#, infos

def rand_neighbor(solution : np.ndarray, action_mask, nb_changes = 1, nb_actions = 2) :
    """
    Generates new random solution.
    :param solution: the solution for which we search a neighbor
    :param nb_changes: maximum number of the changes alowed
    :return: returns a random neighbor for the solution
    """
    new_solution = solution.copy()
    a = np.arange(len(solution), dtype=int)
    i = rd.choice(
        a[action_mask], 
        nb_changes, replace=False
    )
    # if nb_actions == 2:
        # new_solution[i] = 1-new_solution[i]
    # else:
    candidates = list(range(nb_actions))
    candidates.remove(solution[i])
    candidates.remove(0)
    new_solution[i] = rd.choice(candidates, nb_changes, replace=False)
    return new_solution