from copy import deepcopy
from dataclasses import dataclass
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
from numba import njit
from numba.typed import List

from envs.assignment import AssignmentEnv, AssignmentGame, GameEnv
from methods.OR.routing import SA_routing, insertion
# from methods import VA_SA

from typing import Any, Dict, Optional
from time import time
import gymnasium as gym

import pickle

@njit
def knn(a, k):
    idx = np.argpartition(a, k)
    return a[idx[:k]]


class DynamicQVRPEnv(gym.Env):
    
    def __init__(self, 
                #  env : AssignmentEnv = None,
                 K = 50,
                 Q = 50,
                 DoD = 0.,
                 vehicle_capacity = 15, # We assume it homogeneous for all vehicles
                 retain_rate = 0.,
                 use_dataset = True,
                #  is_0_allowed = True,
                 re_optimization  = False,
                 costs_KM = [1], 
                 emissions_KM = [.3], 
                 k_min : int = 2,
                 k_med : int = 5
                 ):
        
        self.instance = -1
        
        if use_dataset:
            retain_comment = f"_retain{retain_rate}" if retain_rate else ""
            # with open(f'data/game_K{K}{retain_comment}.pkl', 'rb') as f:
            #     g = pickle.load(f)
            routes = np.load(f'data/routes_K{K}{retain_comment}.npy')
            all_dests = np.load(f'data/destinations_K{K}{retain_comment}.npy')
            
            if K == 20:
                qs = np.load(f'data/quantities_K{K}_retain1.0.npy')
            else:
                qs = np.ones((len(all_dests), K))
                
            g = AssignmentGame(K=K, Q=Q, costs_KM=costs_KM, emissions_KM=emissions_KM, max_capacity=vehicle_capacity, dynamic=True)
            self._env = GameEnv(AssignmentEnv(game = g, saved_routes = routes, saved_dests=all_dests, saved_q = qs,
                        obs_mode='excess', 
                          change_instance = True, instance_id = self.instance+1), True)
        else:
            self._env = GameEnv(AssignmentEnv(
                AssignmentGame(K=K, Q=Q, dynamic=True))
            , True
            )
        
        self.H = int(DoD*K) # ou = self.K
        self.K = K
        self.p = self._env._env._game.prob_dests
        self.D = self._env._env._game.distance_matrix
        # self.omission_cost = self._env.omission_cost
        # self.CO2_penalty = self._env.CO2_penalty
        self.Q = self._env.Q
        self.k_min = k_min
        self.k_med = k_med
        self.emissions_KM = emissions_KM
        self.costs_KM = costs_KM
        self.max_capacity = vehicle_capacity
        self.remained_capacity = vehicle_capacity*len(costs_KM)
        self.total_capacity = vehicle_capacity*len(costs_KM)
        # self.is_0_allowed = is_0_allowed
        self.num_actions = len(self.emissions_KM) + 1 #if is_0_allowed else len(self.emissions_KM)
        self.re_optimization = re_optimization
        
        self.coordx = self._env.coordx
        self.coordy = self._env.coordy
        
        self.observation_space = gym.spaces.Box(0, 1, (6,)) #TODO * Change if obs change
        self.action_space = gym.spaces.Discrete(2)
        
        
    def _init_instance(self, instance_id):
        self._env.reset(calculate_routes=False)
        
        self.h = 0 # ou = self.K - int(DoD*K)
        if instance_id < 0:
            self.instance += 1
        else:
            self.instance = instance_id
            
        self.dests = self._env.dests
        self.quantities = self._env.quantities
        self.distance_matrix = self._env.distance_matrix
        self.cost_matrix = self._env.cost_matrix
        self.omitted = []
        
        self.episode_reward = 0
        
        # self.dests = self.all_dests[self.instance]
        self.j = self.K - self.H -1
        
        self.action_mask = np.ones(self.K, bool)
        self.action_mask[self.j+1:] = False
        print(self.action_mask)
        print(self.j)
        self.A = np.zeros(len(self.D), bool)
        self.A[self.dests[:self.j]] = True
        self.A[self._env._env._game.hub] = True
        self.NA = ~self.A
        
        self.remained_capacity -= np.sum(self.quantities[:self.j])
        
        
    def _update_instance(self):
        #TODO ***
        pass
    
    def _compute_min_med(self):
        min_knn = np.mean(knn(self.p[self.A]*self.D[self.A, self.j], self.k_min))
        med_knn = np.median(knn(self.p[self.NA]*self.D[self.NA, self.j], self.k_med + 1)[1:])
        
        return min_knn, med_knn
    
    def reset(self, instance_id = -1):
        
        
        self._init_instance(instance_id)
        
        self.assignment, self.routes, info = SA_routing(self._env, self.action_mask)
        
        min_knn, med_knn = self._compute_min_med()
        
        obs = np.array([
            self.quantities[self.j]/ self.total_capacity, # the quantity rate asked by the current demand
            self.remained_capacity / self.total_capacity, # the percentage of capacity remained
            (self.H - self.h) / max(1, self.H), # the remaining demands to come
            min_knn/np.max(self.D), # The mean of the k nearest neighbors in admitted dests
            med_knn/np.max(self.D), # The mean of the k nearest neighbors in non activated dests
            max(0, -info["excess_emission"])/self.Q
            #TODO * Maybe find a better observations
        ])
        
        info.update({
            "episode rewards" : self.episode_reward,
            "quantity accepted" : self.total_capacity - self.remained_capacity,
            "remained capacity" : self.remained_capacity,
        })
        
        return obs, info
    
    #TODO ***
    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        
        self.action_mask[self.dests[self.H + self.h:]] = True
        self.h += 1
        
        self.assignment, self.routes, info = SA_routing(self._env, self.action_mask)
        
        info = dict(info) # It changes ir from the numba dict type
        # info['LCF'] = np.concatenate([[0], costs + emissions*self.CO2_penalty])
        # info['GCF'] = np.sum(info['LCF'])
        
        # r = obs + np.maximum(0, info['LCF'][action] - info['GCF']/self.K) + (action == 0)*self.omission_cost
        if not self.is_0_allowed:
            action -= 1
        
        # normalizer_const = self.K*self.omission_cost
            
        info.update({
            "episode rewards" : self.episode_reward,
            "quantity accepted" : self.total_capacity - self.remained_capacity,
            "remained capacity" : self.remained_capacity,
        })
        
        trunc = self.h >= self.H
        done = trunc or self.Q - total_emission <= + 1e-5
        # info['r'] = np.clip((normalizer_const + info['r'])/normalizer_const, 0, 1)
        
        return obs, r, done, trunc, info
    
    
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