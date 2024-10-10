from copy import deepcopy
from dataclasses import dataclass
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
from numba import njit
from numba.typed import List
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn import preprocessing

from DynamicQVRP.envs.assignment import AssignmentEnv

from typing import Any, Dict, Optional
from time import time
import gymnasium as gym


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
    info['LCF'] = np.zeros(len(cost_matrix), np.float64)
    obs = np.zeros(action.shape, np.float64)
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
                    info['LCF'][i-1] += np.sum(quantities[np.array(alpha)-1])*omission_cost
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
                if k > 1:
                    obs[routes[i-1][k-1] - 1] = cost_matrix[i-1, routes[i-1][k-2], routes[i-1][k-1]] + \
                        cost_matrix[i-1, routes[i-1][k-1], routes[i-1][k]] - \
                        cost_matrix[i-1, routes[i-1][k-2], routes[i-1][k]]
                
                k+=1
                
            costs[i-1] += distance_matrix[routes[i-1][k-1], 0]*costs_KM[i-1]
            emissions[i-1] += distance_matrix[routes[i-1][k-1], 0]*emissions_KM[i-1]
            info['LCF'][i-1] += cost_matrix[i-1, routes[i-1][k-1], 0]
            # routes[i].append(0)
            if k > 1:
                obs[routes[i-1][k-1] - 1] = cost_matrix[i-1, routes[i-1][k-2], routes[i-1][k-1]] + \
                        cost_matrix[i-1, routes[i-1][k-1], routes[i-1][k]] - \
                        cost_matrix[i-1, routes[i-1][k-2], routes[i-1][k]]
                
        # else:
        #     info['omitted'] = alpha
            
    return routes, a, obs, costs, emissions, info
    
    
class DynamicQVRPEnv(gym.Env):
    #TODO
    def __init__(self, 
                 env : AssignmentEnv = None,
                 is_0_allowed = False,
                #  saved_routes = None,
                #  saved_dests = None,
                #  change_instance = True,
                #  instance_id = 0,
                 ):
        
        self._env = env
        self.K = self._env._game.num_packages
        self.omission_cost = self._env._game.omission_cost
        self.CO2_penalty = self._env._game.CO2_penalty
        self.Q = self._env._game.Q
        self.emissions_KM = self._env._game.emissions_KM
        self.costs_KM = self._env._game.costs_KM
        self.max_capacity = self._env._game.max_capacity
        self.is_0_allowed = is_0_allowed
        self.num_actions = len(self.emissions_KM) + 1 if is_0_allowed else len(self.emissions_KM)
        
        self.coordx = self._env._game.coordx
        self.coordy = self._env._game.coordy
        # self.instance_id = instance_id
        # self.saved_routes = saved_routes
        # self.saved_dests = saved_dests
        # self.change_instance = change_instance
        
    
    def reset(self):
        res = self._env.reset()
        self.dests = self._env.destinations
        self.routes = np.zeros((len(self.emissions_KM), self.max_capacity+2), dtype=np.int64)
        for i in range(len(self._env.initial_routes)):
            k = 1
            for j in range(2, len(self._env.initial_routes[i]), 2):
                if not self._env.initial_routes[i, j]:
                    break
                self.routes[i, k] = self._env.initial_routes[i, j]
                k += 1
            
        self.quantities = self._env.quantities
        self.distance_matrix = self._env.distance_matrix
        self.cost_matrix = self._env.costs_matrix
        self.omitted = []
        
        return res
    
    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        
        self.routes = np.zeros((len(self.emissions_KM), self.max_capacity+2), dtype=np.int64)
        self.routes, a, obs, costs, emissions, info = _step(
            action,
            self.routes,
            self.cost_matrix,
            self.distance_matrix,
            self.quantities,
            self.is_0_allowed,
            self.max_capacity,
            self.omission_cost,
            self.costs_KM,
            self.emissions_KM,
        )
        info = dict(info) # It changes ir from the numba dict type
        # info['LCF'] = np.concatenate([[0], costs + emissions*self.CO2_penalty])
        # info['GCF'] = np.sum(info['LCF'])
        
        # r = obs + np.maximum(0, info['LCF'][action] - info['GCF']/self.K) + (action == 0)*self.omission_cost
        if not self.is_0_allowed:
            action -= 1
        
        if np.max(info['LCF']) == np.min(info['LCF']) :
            r = obs/(np.max(obs)*self.quantities) + 1
        else:
            r = obs/(np.max(obs)*self.quantities) + (info['LCF'][action] - np.min(info['LCF']))/(np.max(info['LCF']) - np.min(info['LCF']))
            if np.isnan(r).any():
                print(self._env.reset_counter)
                print('obs : ', obs)
                print('info[LCF] : ', info['LCF'])
                print('q : ', self.quantities)
                print('a : ', action)
        
        # normalizer_const = self.K*self.omission_cost
            
        total_emission = np.sum(emissions)
        info['r'] = -(np.sum(costs) + max(0, total_emission - self.Q - 1e-5)*self.CO2_penalty + np.sum(a == 0)*self.omission_cost)
        info['a'] = a
        info['routes'] = self.routes
        info['costs per vehicle'] = costs
        info['omitted'] = np.where(a==0)[0]
        info['excess_emission'] = total_emission - self.Q - 1e-5
        self.omitted = info['omitted']
        self.obs = obs
        # info['r'] = np.clip((normalizer_const + info['r'])/normalizer_const, 0, 1)
        
        return obs, r, total_emission <= self.Q + 1e-5, False, info
    
    
    def render(self):
        G = nx.DiGraph()
        G.add_nodes_from(list(range(len(self.dests))))
        node_attrs = dict()
        edges = []
        # vehicle_edges = []
        print(self.routes.shape)
        for o in self.omitted:
            node_attrs[o+1] = {
                         'vehicle' : 0, 
                        #  'x' : [self.dests[int(self.routes[m, j])], m, gained_on_substitution], 
                         'pos' : (self.coordx[self.dests[o]], self.coordy[self.dests[o]]),
                         'q' : 50*self.quantities[o],
            }
        
        for m in range(len(self.routes)):
            j = 1
            # vehicle_edges.append([])
            while not (j > 0 and self.routes[m, j] == 0):
                # if self.routes[m, j] == 0:
                #     gained_on_substitution = 0.
                # else:
                #     #TODO
                #     gained_on_substitution = self.routes[m, j-1] + self.routes[m, j+1] -(
                #             self.cost_matrix[m, self.dests[int(self.routes[m, j-2])], self.dests[int(self.routes[m, j+2])]]
                #         )
                node_attrs[int(self.routes[m, j])] = {
                         'vehicle' : m+1, 
                        #  'x' : [self.dests[int(self.routes[m, j])], m, gained_on_substitution], 
                         'pos' : (self.coordx[self.dests[int(self.routes[m, j])-1]], self.coordy[self.dests[int(self.routes[m, j])-1]]),
                         'q' : 50*self.quantities[self.routes[m, j]-1],
                }
                # G_ncolors[int(route[m, j])] = colors[m]
                # if int(route[m, j]):
                #     G_ncolors.append(colors[m])
                # G_pos[int(route[m, j])] = (dest[int(route[m, j])]//15, dest[int(route[m, j])]%15)
                # print(int(route[m, j]), int(route[m, j+2]))
                edges.append(
                    (
                        int(self.routes[m, j-1]),
                        int(self.routes[m, j]),
                        self.cost_matrix[m, int(self.routes[m, j-1]), int(self.routes[m, j])]
                    )
                )
                # vehicle_edges[m].append((int(route[m, j]), int(route[m, j+2])))
                j+=1
            if int(self.routes[m, j-1]) != int(self.routes[m, j]):
                edges.append(
                        (
                            int(self.routes[m, j-1]),
                            int(self.routes[m, j]),
                            self.cost_matrix[m, int(self.routes[m, j-1]), int(self.routes[m, j])]
                        )
                    )
        node_attrs[0] = {
                         'vehicle' : 0, 
                        #  'x' : [self.dests[int(self.routes[m, j])], m, gained_on_substitution], 
                         'pos' : (self.coordx[self._env._game.hub], self.coordy[self._env._game.hub]),
                         'q' : 200
                }
        G.add_weighted_edges_from(edges)
        # print(G.nodes)
        nx.set_node_attributes(G, node_attrs)
        
        colors = []#'#1f78b4' for _ in range(len(G.nodes))]
        colors.append('lightgray')
        colors.append('lightgreen')
        if len(self.emissions_KM) > 3:
            colors.append('lightblue')
        colors.append('lightyellow')
        colors.append('lightcoral')#'red')
        G_ncolors = [colors[m] for m in nx.get_node_attributes(G,'vehicle').values()]
        G_ncolors[0] = 'gray'

        _, ax = plt.subplots(figsize=(10, 7))
        weights = list(nx.get_edge_attributes(G,'weight').values())
        nx.draw_networkx(G, 
                         pos = nx.get_node_attributes(G,'pos'),  
                         ax=ax, 
                         font_size=5, 
                         with_labels=True,
                         node_size=list(nx.get_node_attributes(G,'q').values()), 
                         node_color=G_ncolors,
                         edge_color = weights,
                         edge_cmap=plt.cm.jet,
                         node_shape='s',
                         arrows=True
        )

        # handles, labels = ax.get_legend_handles_labels()
        # labels = list(range(len(colors)))
        ax.scatter([0],[0],color=colors[0],label=f'Omitted')
        for i in range(1, len(colors)):
            ax.scatter([0],[0],color=colors[i],label=f'Vehicle {i}')
        ax.scatter([0],[0],color='white')

        # reverse the order
        plt.draw()
        plt.legend(bbox_to_anchor=(1.4, 1.0), loc='upper right')
        mesh = ax.pcolormesh(([], []), cmap = plt.cm.jet)
        mesh.set_clim(np.min(weights),np.max(weights))
        # Visualizing colorbar part -start
        plt.colorbar(mesh,ax=ax)
        # plt.colorbar()
        # plt.style.use("dark_background")
        # plt.legend()
        plt.show()

        return nx.to_latex(G, nx.get_node_attributes(G,'pos'), node_options=dict(zip(range(len(G_ncolors)), G_ncolors)))