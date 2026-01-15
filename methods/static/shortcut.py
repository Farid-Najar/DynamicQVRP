from dataclasses import dataclass
import sys
import os
from copy import deepcopy
# from SA_baseline import recuit
# direc = os.path.dirname(__file__)
# pri&
# caution: path[0] is reserved for script path (or '' in REPL)
# print(str(path)+'/ppo')
sys.path.insert(1, '/Users/faridounet/PhD/TransportersDilemma')
JIT = False
import numpy as np
if JIT:
    from numba import njit
else:
    njit = lambda x : x
from numba.typed import List

@dataclass
class Table:
    values : np.ndarray
    sol : np.ndarray
    max_omitted : np.int64
    
@njit
def compute_delta(cost_matrix, route, k):
    #compute the difference in cost when removing elements in the tour
    delta = np.zeros((len(route), k+1), dtype=np.float64)#malloc(t.length*sizeof(float*));
    
    # observation[int(self.initial_routes[m, j])-1] = self.initial_routes[m, j-1] + self.initial_routes[m, j+1] -(
    #                             costs_matrix[m, int(self.initial_routes[m, j-2]), int(self.initial_routes[m, j+2])]
    #                     )
    #no need for delta[0]
    for i in range(2, len(route)):
        current_vertex = route[i]
        sum = cost_matrix[current_vertex, route[i-1]]
        # //printf("\n Removing before vertex %d :",i);
        for j in range(1, min(k+1,i)):
            if(route[i-j]):#//the vertex is not 0, hence can be removed
                sum += cost_matrix[route[i-j-1], route[i-j]]
                delta[i, j] = sum - cost_matrix[route[i-j-1], current_vertex]
                # //printf("(%d,%f)  ", j ,delta[i][j]);
            else:#//the vertex is 0, we cannot remove
                delta[i][j] = -1 #//special value to escape from the computation
                break
    #print(delta)
    return delta


@njit
def compute_smallest_cost(cost_matrix, route, q): 
    K = len(route) - 1
    cum_q = np.cumsum(q)
    
    delta = compute_delta(cost_matrix, route, K+1)
    #print("route",route)
    #print("cost_mat",cost_matrix)
    # print("delta",delta)
    # values, sol = init_table(len(route),K+1)
    sol = np.zeros((K, len(route)), dtype=np.int64)
    values = np.zeros((K, len(route)))
    # max_omitted = K #quantity max that need to be ommited to satisfy the constraint
    for k in range(1, K):
        # if values[k-1, len(route)-1] >= excess:
        #     # while the pollution constraint is violated, try to remove one more in quantity
        #     # max_omitted = k
        #     break
        
        debut = np.argmin(cum_q < k)# +1
        for i in range(debut, len(route)):
            # we begin from debut, the first index from which we 
            # are able to remove quantity k before it
            # 
            # j is the number of consecutive elements we remove before i
            for j in range(i): 
                #break when an element cannot be removed
                if delta[i][j] == -1:
                    break
                oq = q[i-j: i].sum() # when it sums the empty slice, it is zero
                if k>=oq: #it is possible to omit thiq quantity without removing more than k
                    val = delta[i, j] + values[k-oq, i-j-1]
                    if(val > values[k, i]):
                        values[k, i] = val
                        sol[k, i] = j
                else:
                    break
        
    # // //print for debugging
    # // for(int i = 0; i <= tab.max_omitted; i++){
    # //     printf("\n Number of ommited packages %d, gain %f",i,tab.values[i][t.length -1]);
    # // }
    # // int *s = get_solution(tab,t.length);
    # // printf("Position of packages omitted in the solution: ");
    # // for(int i = 0; i < tab.max_omitted; i++){
    # //     printf("%d ",s[i]);
    # // }
    return sol, values#, max_omitted

@njit
def value(value_tables, coeff, types, sol, excess):
    #evaluate the value and pollution of a solution
    gain = 0.
    pol = 0.
    for i in range(types):
        gain += value_tables[i][sol[i]]
        pol += value_tables[i][sol[i]]*coeff[i]
    # print('gain : ', gain)
    # print('pol : ', pol)
    # print("excess ", pol - excess)
    return gain if pol - excess >= -1e-5 else 0.

@njit
def best_combination(k, types, current_type, value_tables, #max_omitted
                     coeff, excess, sol, max_val, max_sol):
    #extremly simplistic enumeration of the way to generate k
    #we could also determine by dynamic programming all value of k rather than testing every possible value
    total = sol[:current_type].sum()
    # print('best', total)
    # print('current type', current_type)
    # print('current type', types)
    #printf("Total %d current type %d\n",total,current_type);
    if(current_type == types -1):   
        if k - total > len(value_tables[0])-1:#max_omitted[current_type]:
            # print(k, total, len(sol))
            # print(len(value_tables[0]))
            return #not a possible solution
        sol[current_type] = k - total
        # print('sol : ', sol)
        val = value(value_tables,coeff,types,sol,excess)
        # print("value : ",val)
        if (val > max_val[0]):
            max_val[0] = val
            # print('max val : ', max_val)
            max_sol[:] = sol[:]
            # for i in range(sol.shape[0]):
            #     max_sol[i] = sol[i]
            #max_sol =  sol
            # print("solution trouvée : ",max_sol,"k : ",k,"index actif",current_type)    
    else:    
        for i in range(min(k - total + 1,len(value_tables[0]))):
            sol[current_type] = i
            best_combination(k, types, current_type + 1, value_tables,coeff, excess, sol, max_val, max_sol)

@njit
def get_solution_single_type(sol, k, tour_length, q):
    #get an optimal solution with k omitted packages from a full dynamic table
    # solution = np.zeros(k, dtype=np.int64)
    l = List()
    position = tour_length-1
    #printf("\n K max considéré: %d Position max : %d\n",k,position);
    while(k):
        #continue until it has found all packets to remove
        to_remove = sol[k][position]
        #printf("%d %d %d\n",position,k,to_remove);
        k -= q[position-to_remove: position].sum() 
        l += List(range(position-1, position-to_remove-1, -1))
        # print(to_remove)
        # print(position-1, position-to_remove+1)
        # for i in range(1, to_remove+1):
            # k -= 1
            # solution[k] = position -i
        
        position -= to_remove+1
    solution = np.array(l)
    return np.flip(solution)


@njit
def get_solution_multiple_types(sol, max_sol, routes, types, q):
    #get a solution from a full dynamic table
    solution = []#np.zeros((types, len(routes[0])), dtype=np.int64)
    for i in range(types):
        solution.append(get_solution_single_type(sol[i],max_sol[i],len(routes[i]), q[i]))
    return solution
    
@njit
def multi_types(cost_matrix, rtes, coef, excess, quants):
    
    selection = np.where(coef > 0)[0]
    
    routes = rtes[selection]
    coeff  = coef[selection]
    q  = quants[selection]
    # print('qs : ', q)
    
    types = len(routes)
    K = len(routes[0])-2 # np.amax(np.sum(q, 1))
    
    #types is the number of types (and thus of tour and coeff)

     #problem here plutot une liste de tableau qui peuvent avoir des tailles différentes
    sol = np.zeros((types, K+1, len(routes[0])), dtype=np.int64)
    values = np.zeros((types, K+1, len(routes[0])), dtype=float)
    # max_omitted = np.zeros(types, dtype=np.int64)
    
    #weight = coeff/np.sum(coeff)
    
    value_tables = []#List()
    # print('excess/coeff : ', excess/coeff)
    
    for i in range(types):
        sol[i], values[i] = compute_smallest_cost(cost_matrix, routes[i], q[i])
        #extract the best combination of omission between the different types
        value_tables.append(values[i, :, -1])
        
    # print('value tables : ', value_tables)
    # # print('max omitted tables : ', max_omitted)
    # print('sol tables : ', sol)
    # print(100*'-')
    
    # print(values[i, 1])
    # print(values[i, 1, -1])
    # print("taille des routes")
    # for i in range(types):
    #     print(i," : ", len(routes[i]),"\n")
    #print("excess",excess, "value_tables:",value_tables)

    solution = np.zeros(types, dtype=np.int64)
    max_sol = np.zeros(types, dtype=np.int64)
    max_val = np.zeros(1)
    # print("max ommited:",max_omitted)
    for k in range(1, types*(K+1)):#max_omitted.sum() +1):
        #we could begin with a larger k, compute by how much
        #printf("k: %d \n",k);
        best_combination(k, types, 0, value_tables, coeff, excess, solution, max_val, max_sol)
        if(max_val[0] != 0):
            # print("solution de taille",k,max_sol,"valeur",max_val[0])
            break
        
    # print('value tables : ', value_tables)
    # # print('max omitted tables : ', max_omitted)
    # print('sol tables : ', sol)
    # print('max sol tables : ', max_sol)
    
    final_sol = get_solution_multiple_types(sol, max_sol, routes, types, q)
    # printf("best solution of value: %f\n",*max_val);
    # for i in range(types):
    #     print(max_sol[i], "packages omitted in tour ", i, " : ", end='')
    #     for j in range(max_sol[i]):
    #         print(final_sol[i][j]," ", end='')
    #     print()
        
    # print(final_sol)
    a = np.array([
        routes[i, final_sol[i][j]]-1
        for i in range(len(final_sol))
        for j in range(len(final_sol[i]))
    ], dtype=np.int64)#np.zeros(len(cost_matrix)-1)
    
    # print(value_tables)
    
    # for i in range(types):
    #     for j in range(max_sol[i]):
    #         a[solution[i][j]] = 1.
            
    return a#, max_val, max_sol


# if __name__ == '__main__':
#     # import pickle
#     # try:
#     #     with open(f'TransportersDilemma/RL/game_K1000.pkl', 'rb') as f:
#     #         g = pickle.load(f)
#     # except:
#     #     g = AssignmentGame(grid_size=35, K = 1000, Q = 500, max_capacity=250)
#     #     with open(f'./game_K1000.pkl', 'wb') as f:
#     #         pickle.dump(g, f, -1)
            
#     # print('The game is ready!')
#     import pickle
#     real = "real_"
#     K = 20
#     retain = 1.0
#     retain_comment = f"_retain{retain}"if retain else ""

#     with open(f'{real}res_compare_EG_A*_SA_K50_n100.pkl', 'rb') as f:
#         data = pickle.load(f)

#     with open(f'RL/{real}game_K{K}{retain_comment}.pkl', 'rb') as f:
#         g = pickle.load(f)
#     routes = np.load(f'RL/{real}routes_K{K}{retain_comment}.npy')
#     dests = np.load(f'RL/{real}destinations_K{K}{retain_comment}.npy')
#     if K == 20:
#         qs = np.load(f'RL/{real}quantities_K{K}{retain_comment}.npy')

#     env = RemoveActionEnv(game = AssignmentGame(
#         K = 5, Q = 30, max_capacity=10, real_data=True, emissions_KM=[.3], costs_KM=[1]
#         ),
#     )
#     idx  = 11
#     env = RemoveActionEnv(game = g, saved_routes = routes, saved_dests=dests, 
#                       action_mode = 'destinations', saved_q = qs if K == 20 else None, 
#                         change_instance = False, rewards_mode='normalized_terminal', instance_id = idx)
    
#     # quantities = np.ones(5, dtype=int)
#     # C = env._env._game.max_capacity * env._env._game.num_vehicles - env._env._game.num_packages
#     # c = (C*np.random.dirichlet(np.ones(5))).astype(int)
#     # quantities += c
#     # from assignment import Package
#     # packages = [
#     #         Package(
#     #             destination=k+1,
#     #             quantity=quantities[k],
#     #         )
#     #         for k in range(5)
#     #     ]
#     # obs, info = env.reset(packages = packages, time_budget = 1)
#     obs, info = env.reset()
#     # print(env._env.distance_matrix*.3)
    
#     # print(info['excess_emission'])
#     routes = np.array([
#         [
#             env._env.initial_routes[m, i] 
#             for i in range(0, len(env._env.initial_routes[m]), 2)
#         ]
#         for m in range(len(env._env.initial_routes))
#     ], dtype=np.int64)
    
#     q = np.array([
#         [
#             env._env.quantities[i-1] if i else 0
#             for i in routes[m]
#         ]
#         for m in range(len(routes))
#     ], dtype=np.int64)
#     print(routes)
#     env_SA = deepcopy(env)
#     print('gains : ', )
#     action_SA, *_ = recuit(deepcopy(env_SA._env), 5000, 1, 0.9999, H=100_000)
#     print('sa : ', len(action_SA) - np.sum(action_SA))
#     print('sa : ', action_SA)
#     print('excess : ', info['excess_emission'])
#     ee = info['excess_emission']
#     info_SA = info
#     a_SA = np.where(action_SA == 0)[0]
#     for aa in a_SA:
#         # print(f'obs : {obs[:-100].astype(int)}')
#         print(100*'-')
#         obs, r_SA, *_, info_SA = env_SA.step(aa)
#         print('removed ', aa+1, ', gained : ', ee - info_SA['excess_emission'])
#         ee = info_SA['excess_emission']
        
#         # rtes = np.array([
#         # [
#         #     env._env.initial_routes[m, i] 
#         #     for i in range(0, len(env._env.initial_routes[m]), 2)
#         # ]
#         # for m in range(len(env._env.initial_routes))
#         # ], dtype=int)
#         # print(rtes)
#     print('excess : ', info_SA['excess_emission'])
#     print('r_SA : ', r_SA)
    
#     print(a_SA + 1)
#     print()
#     print(50*'-')
#     print('SHORTCUT')
#     print()
#     print(info['excess_emission'])
    
#     coeff = env._env._game.emissions_KM
#     # CM = np.array([
#     #     env._env.distance_matrix*coeff[i]
#     #     for i in range(len(coeff))
#     # ]).copy()
    
#     # assert (CM[1] == env._env.distance_matrix*.3).all()
    
#     rtes = routes.copy()
#     print(env._env.distance_matrix.shape)
#     a = multi_types(env._env.distance_matrix, routes, coeff, info['excess_emission'], q)
#     print(a+1)
#     print(np.where(action_SA == 0)[0]+1)
#     ee = info['excess_emission']
#     print('tot emission : ', ee)
#     cum_ee = 0.
#     for aa in a:
#         idxi, idxj = np.where(rtes == aa+1)
#         # idxi = idxi[0]
#         # idxj = idxj[0]
#         # i0 = rtes[idxi, idxj-1]
#         # i1 = rtes[idxi, idxj]
#         # i2 = rtes[idxi, idxj+1]
#         # x0 = CM[idxi, i0, i1]
#         # x1 = CM[idxi, i1, i2]
#         # x2 = CM[idxi, i0, i2]
#         # diff = CM[idxi, i0, i1] + CM[idxi, i1, i2] - CM[idxi, i0, i2]
#         # for ii in range(idxj, len(rtes[idxi])-1):
#         #     rtes[idxi, ii] = rtes[idxi, ii+1]
#         _, r, *_, inf = env.step(aa)
#         eee = ee - inf['excess_emission']
#         # assert abs(eee - diff) < 1e-10
#         cum_ee += eee
#         print('removed ', aa, ', gained : ', eee)
#         print('cumulated : ', cum_ee)
#         print()
#         ee = inf['excess_emission']
#     # _, r, *_ = env.step(a)
#     print(r)
#     print(r_SA)
#     print(r_SA<=r)
    
    