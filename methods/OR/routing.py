from typing import Dict, List
import numpy as np
# from envs.assignment import AssignmentEnv, GameEnv

#import itertools as it
import multiprocess as mp
from numpy import random as rd
from numpy import exp
from copy import deepcopy
from numba import njit
from numba.typed import List
from methods.OR.metaheuristics import SA_vrp

# from envs import DynamicQVRPEnv

@njit
def NN_routing(
    action,
    routes,
    cost_matrix,
    distance_matrix,
    quantities,
    max_capacity,
    costs_KM,
    emissions_KM,
):
    # routes = []#List()
    costs = np.zeros(len(cost_matrix), np.float64)
    emissions = np.zeros(len(cost_matrix), np.float64)
    # info['LCF'] = np.zeros(len(cost_matrix), np.float64)
    # info['omitted'] = alpha    
    
    a = action.copy()
    
    alphas = [list(np.where(a == v)[0] + 1) for v in range(1, len(cost_matrix)+1)]
    vs = np.argsort(np.array([len(l) for l in alphas]))[::-1]
    # print(alphas)
    # print(vs)
    
    for i in range(len(vs)):
        # routes.append([])#List([0]))
        # if v:
        v = vs[i] + 1
        alpha = alphas[v-1]
        quantity = 0
        # routes[i].append(int(0))
        k = 1
        while True:
            if len(alpha) == 0:
                break
            j = int(np.argmin(cost_matrix[v-1, routes[v-1][k-1], np.array(alpha)]))
            
            quantity += quantities[alpha[j]-1]
            
            if quantity > max_capacity:
                # routes[0] += alpha
                # info['LCF'][i-1] += np.sum(quantities[np.array(alpha)-1])*omission_cost
                if i < len(vs)-1:
                    a[np.array(alpha) - 1] = vs[i+1] + 1
                    alphas[vs[i+1]].extend(alpha)
                    # action[np.array(alpha) - 1] += 1
                else:
                    # a[np.array(alpha) - 1] = 0
                    raise(Exception("Something is going wrong. This function (NN_routing) cannot calculate routes."))
                    # action[np.array(alpha) - 1] = 0
                # print(np.array(alpha), "omitted")
                break
            # temp = cost_matrix[i-1, routes[i-1][-1], alpha]
            # print(cost_matrix[i-1, routes[i-1][-1], np.array(alpha)])
            # print(alpha)
            dest = alpha.pop(j)
            # if k <= max_capacity:
            costs[v-1] += distance_matrix[routes[v-1][k-1], dest]*costs_KM[v-1]
            emissions[v-1] += distance_matrix[routes[v-1][k-1], dest]*emissions_KM[v-1]
            # info['LCF'][i-1] += cost_matrix[i-1, routes[i-1][k-1], dest]
            routes[v-1, k] = dest
            # print(routes[i-1], costs[i-1], emissions[i-1])
            # if k > 1:
            #     obs[routes[i-1][k-1] - 1] = cost_matrix[i-1, routes[i-1][k-2], routes[i-1][k-1]] + \
            #         cost_matrix[i-1, routes[i-1][k-1], routes[i-1][k]] - \
            #         cost_matrix[i-1, routes[i-1][k-2], routes[i-1][k]]
            
            k+=1
            
        costs[v-1] += distance_matrix[routes[v-1][k-1], 0]*costs_KM[v-1]
        emissions[v-1] += distance_matrix[routes[v-1][k-1], 0]*emissions_KM[v-1]
        # info['LCF'][i-1] += cost_matrix[i-1, routes[i-1][k-1], 0]
        # routes[i].append(0)
        # if k > 1:
        #     obs[routes[i-1][k-1] - 1] = cost_matrix[i-1, routes[i-1][k-2], routes[i-1][k-1]] + \
        #             cost_matrix[i-1, routes[i-1][k-1], routes[i-1][k]] - \
        #             cost_matrix[i-1, routes[i-1][k-2], routes[i-1][k]]
                
        # else:
        #     info['omitted'] = alpha
            
    return routes, a, costs, emissions

#TODO ** Use a better routing
# The current routing (NN routing) is not very efficient.

def _run(env, assignment):
    
    routes = np.zeros((len(env.emissions_KM), env.max_capacity+2), dtype=np.int64)
    # print([np.sum(env.quantities[assignment == v]) > env.max_capacity for v in range(1, len(env.costs_KM)+1)])
    if (np.sum(env.quantities[assignment.astype(bool)]) > env.total_capacity
        or np.any([np.sum(env.quantities[assignment == v]) > env.max_capacity for v in range(1, len(env.costs_KM)+1) ])
        ):
        return -env.K*env.omission_cost, False, env.info
    
    routes, a, costs, emissions = NN_routing(
        assignment,
        routes,
        env.cost_matrix,
        env.distance_matrix,
        env.quantities,
        env.max_capacity,
        List(env.costs_KM),
        List(env.emissions_KM),
    )
    
    total_emission = np.sum(emissions)
    info = dict()
    info['assignment'] = a
    info['routes'] = routes
    info['costs per vehicle'] = costs
    info['omitted'] = np.where(a==0)[0]
    info['remained_quota'] = env.Q - total_emission
    
    env.routes = routes
    
    r = -(np.sum(costs) + max(0, total_emission - env.Q - 1e-5)*env.CO2_penalty + np.sum(a == 0)*env.omission_cost)
    d = total_emission - env.Q - 1e-5 <= 0
    # r *= float(d)
    
    return r, d, info
    
    
def insertion(env, action = None):
    
    assignment = env.assignment
    best = 0
    r, d, info = _run(env, assignment)
    eval_best = r*float(d)
    best_info = deepcopy(info)
    best_routes = env.routes.copy()
    
    if action is not None:
        assignment[env.j] = action
        _, d, info = _run(env, assignment)
        if d:
            best_info = deepcopy(info)
            best_routes = env.routes.copy()
        else:
            assignment[env.j] = 0
        
        return assignment, best_routes, best_info
    
    for v in range(1, len(env.costs_KM) + 1):
        assignment[env.j] = v
        r, d, info = _run(env, assignment)
        
        if r > eval_best and d:
            eval_best = r
            best = v
            best_info = deepcopy(info)
            best_routes = env.routes.copy()
            
    assignment[env.j] = best
    return assignment, best_routes, best_info
        
@njit
def construct_initial_solution(j, q, a, V, max_capacity):
    # j, assignment, V, max_capacity
    
    assignment = np.zeros_like(a)
    remained_cap = List([float(max_capacity) for _ in range(V)])
    # remained_cap = [max_capacity for _ in range(V)]
    for i in range(j):
        for v in range(1, V + 1):
            if remained_cap[v-1] - q[i] >= 0:
                remained_cap[v-1] -= q[i]
                assignment[i] = v
                break
    
    return assignment

def construct_emergency_solution(env, j = None):
    # j, assignment, V, max_capacity
    
    assignment = np.zeros_like(env.assignment)
    j = env.j if j is None else j
    # best = assignment.copy()
    # best_info = {}
    remained_cap = [env.max_capacity for _ in range(len(env.costs_KM))]
    _, d, best_info = _run(env, assignment)
    best_routes = env.routes.copy()
    
    for i in range(j):
        for v in range(1, len(env.costs_KM) + 1):
            if remained_cap[v-1] - env.quantities[i] >= 0:
                a = assignment.copy()
                a[i] = v
                _, d, info = _run(env, a)
                if d:
                    remained_cap[v-1] -= env.quantities[i]
                    assignment[i] = v
                    best_info = deepcopy(info)
                    best_routes = env.routes.copy()
                break
            
    return assignment, best_routes, best_info

def construct_greedy_solution(env):
    # j, assignment, V, max_capacity
    #TODO make a greedy initial solution especially for offline
    assignment = np.zeros_like(env.assignment)
    # best = assignment.copy()
    best_info = {}
    remained_cap = [env.max_capacity for _ in range(len(env.costs_KM))]
    for i in range(env.j):
        for v in range(1, len(env.costs_KM) + 1):
            if remained_cap[v-1] - env.quantities[i] >= 0:
                a = assignment.copy()
                a[i] = v
                _, d, info = _run(env, a)
                if d:
                    remained_cap[v-1] -= env.quantities[i]
                    assignment[i] = v
                    best_info = deepcopy(info)
                    best_routes = env.routes.copy()
                break
            
    
    return assignment, best_routes, best_info

def SA_routing(env,
               offline_mode = False, random_start = False,
               T_init = 1_000, T_limit = 1, lamb = .995, var = False, id = 0, log = False, H = np.inf) :
    """
    This function finds a solution for the steiner problem
        using annealing algorithm
    :param game: the assignment game
    :param T_init: the initial temperature
    :param T_limit: the lowest temperature allowed
    :return: the solution found and the evolution of the best evaluations
    """
    
    action_mask = env.action_mask
    num_actions = len(env.costs_KM) + 1
    static_mode = env.H == 0
    is_O_allowed = env.is_O_allowed
    
    #! problem with the initial solution
    # when the offline method is called in the full dyn case,
    # the initial sol is zero but must debute with another one
    # TODO ** fix the initial solution
    if env.h == 0 and env.j:
        best = construct_initial_solution(
            env.j,
            env.quantities,
            env.assignment,
            len(env.costs_KM),
            env.max_capacity
        )
    elif offline_mode:
        # best = construct_initial_solution(
        #     len(env.assignment),
        #     env.quantities,
        #     env.assignment,
        #     len(env.costs_KM),
        #     env.max_capacity
        # )
        best, best_routes, best_info = construct_emergency_solution(env, len(env.assignment))
        best = best_info['assignment']
    else:
        best = env.assignment.copy()
        
    # print(best)
    best[~action_mask & is_O_allowed] = 0
    if not offline_mode:
        best[env.j] = 1 if env.h > 0 or static_mode else 0
    solution = best.copy()
    T = T_init
    best_routes = env.routes.copy()
    best_info = dict()
    r, d, info = _run(env, best)
    if not d:
        best[env.j] = 0
        best_info = deepcopy(env.info)
    else:
        best = info['assignment'].copy()
        best_routes = env.routes.copy()
        best_info = deepcopy(info)
    eval_best = -r
    eval_solution = eval_best
    m = 0
    # list_best_costs = [eval_best]
    flag100 = True
    # infos = []
    init_flag = not d and env.h == 0 and env.allow_initial_omission
    full_dyn_flag = env.h == 0 and env.j==0
    
    if not static_mode:
        if full_dyn_flag or not init_flag  and num_actions <= 2:
            if "remained_quota" not in best_info.keys():
                return construct_emergency_solution(env)
                
            return best, best_routes, best_info
    
    # if static_mode:
    #     num_actions += 1
    
    while(T>T_limit):
        
        # infos['T'].append(T)
        sol = rand_neighbor(solution, action_mask, nb_actions=num_actions, allow_0 = is_O_allowed)
        r, d, info = _run(env, sol)
        eval_sol = -r
        
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
                best = info['assignment'].copy()
                best_routes = env.routes.copy()
                best_info = deepcopy(info)
            eval_best = eval_sol
            # infos.append(info)
            
        if eval_sol < eval_solution :
            prob = 1
        else :
            prob = exp((eval_best - eval_sol)/T)
        
        if rd.random() <= prob :
            solution = sol
            eval_solution = eval_sol
        # list_best_costs.append(eval_best)
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
    
    if "remained_quota" not in best_info.keys():
        return construct_emergency_solution(env)
    
    return best, best_routes, best_info#, list_best_costs#, infos

def rand_neighbor(solution : np.ndarray, action_mask, allow_0, nb_changes = 1, nb_actions = 2) :
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
    if not allow_0[i]:
        candidates.remove(0)
    new_solution[i] = rd.choice(candidates, nb_changes, replace=False)
    return new_solution

def SA_routing2(env,# : DynamicQVRPEnv,
               offline_mode = False,
               T_init = 1_000, T_limit = 1, lamb = .995, log = False, H = 50_000):
    
    
    distance_matrix = env.distance_matrix
    qs = env.quantities
    customers = np.arange(1, env.K + 1)[env.action_mask]
    
    initial_solution = env.routes.flatten()
    initial_solution = initial_solution[initial_solution != 0]
    # print(initial_solution)
    # initial_solution = None
    
    # We adapt the hyper parameters for faster algorithms
    max_iter = min(H, len(customers)*((len(env.emissions_KM)+1)//2)*1000)
    # print(max_iter)
    T_init = min(T_init, len(customers)*100)
    
    total_emissions, oq, routes, assignment = SA_vrp(
        distance_matrix, env.Q, qs[customers], env.max_capacity, env.emissions_KM, 
        customers = customers, initial_solution = initial_solution, log = log,
        SA_configs = dict(
          initial_temp=T_init,
          cooling_rate=lamb,
          max_iter=max_iter, 
        ),
    )
    
    if oq and env.h: # In the dynamic part, the omission is not allowed
        assignment = env.assignment
        routes = env.routes
        info = env.info
    else:
        info = dict()
        info['assignment'] = assignment
        info['routes'] = routes
        # info['costs per vehicle'] = costs
        info['omitted'] = np.where(assignment==0)[0]
        info['remained_quota'] = env.Q - total_emissions
    
    return assignment, routes, info