from copy import deepcopy
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
from numba import njit
from numba.typed import List

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

def load_data():
    coordx = np.load('data/coordsX.npy')
    coordy = np.load('data/coordsY.npy')
    D = np.load('data/distance_matrix.npy')
    probs = np.load('data/prob_dests.npy')
    
    return D, coordx, coordy, probs


class DynamicQVRPEnv(gym.Env):
    
    def __init__(self, 
                #  env : AssignmentEnv = None,
                 K = 50,
                 Q = 50,
                 DoD = 0.,
                 vehicle_capacity = 15, # We assume it homogeneous for all vehicles
                 retain_rate = 0.,
                 use_dataset = True,
                 re_optimization  = False,
                 costs_KM = [1], 
                 emissions_KM = [.3], 
                 CO2_penalty = 10_000,
                 k_min : int = 2,
                 k_med : int = 5,
                 hub = 0
                 ):
        
        self.instance = -1
        
        if use_dataset:
            retain_comment = f"_retain{retain_rate}" if retain_rate else ""
            # with open(f'data/game_K{K}{retain_comment}.pkl', 'rb') as f:
            #     g = pickle.load(f)
            # routes = np.load(f'data/routes_K{K}{retain_comment}.npy')
            self.all_dests = np.load(f'data/destinations_K{K}{retain_comment}.npy').astype(int)
            
            if K == 20:
                qs = np.load(f'data/quantities_K{K}_retain1.0.npy')
            else:
                qs = np.ones((len(self.all_dests), K))
                
            # g = AssignmentGame(K=K, Q=Q, costs_KM=costs_KM, emissions_KM=emissions_KM, max_capacity=vehicle_capacity, dynamic=True)
            # self._env = GameEnv(AssignmentEnv(game = g, saved_routes = routes, saved_dests=all_dests, saved_q = qs,
            #             obs_mode='excess', 
            #               change_instance = True, instance_id = self.instance+1), True)
        else:
            raise("not implemented yet")
            # self._env = GameEnv(AssignmentEnv(
            #     AssignmentGame(K=K, Q=Q, dynamic=True))
            # , True
            # )
        
        self.H = int(DoD*K) # ou = self.K
        self.K = K
        self.D, self.coordx, self.coordy, self.p = load_data()
        
        self.qs = qs
        # self.omission_cost = self._env.omission_cost
        # self.CO2_penalty = self._env.CO2_penalty
        self.Q = Q
        self.hub = hub
        self.k_min = k_min
        self.k_med = k_med
        self.emissions_KM = emissions_KM
        self.costs_KM = costs_KM
        self.max_capacity = vehicle_capacity
        self.total_capacity = vehicle_capacity*len(costs_KM)
        # self.is_0_allowed = is_0_allowed
        self.num_actions = len(self.emissions_KM) + 1 #if is_0_allowed else len(self.emissions_KM)
        self.re_optimization = re_optimization
        
        self.CO2_penalty = CO2_penalty
        self.omission_cost = (2*np.max(self.D) +1)*np.max(self.costs_KM)
        
        self.observation_space = gym.spaces.Box(0, 1, (6,)) #TODO * Change if obs change
        self.action_space = gym.spaces.Discrete(2)
        
        
    def _init_instance(self, instance_id):
        # self._env.reset(calculate_routes=False)
        
        self.h = 0 # ou = self.K - int(DoD*K)
        if instance_id < 0:
            self.instance += 1
        else:
            self.instance = instance_id
            
        self.dests = self.all_dests[self.instance]
        self.quantities = self.qs[self.instance]
        
        self.remained_capacity = self.total_capacity
        
        l = [self.hub] + list(self.dests)
        self.mask = np.ix_(l, l)
        self.distance_matrix = self.D[self.mask]
        self.cost_matrix = np.array([
            (self.costs_KM[v] + self.CO2_penalty*self.emissions_KM[v])*self.distance_matrix
            for v in range(len(self.costs_KM))
        ])
        self.omitted = []
        
        self.episode_reward = 0
        
        # self.dests = self.all_dests[self.instance]
        self.j = self.K - self.H
        self.assignment = np.ones(self.K, int)
        
        self.action_mask = np.ones(self.K, bool)
        self.is_O_allowed = np.zeros(self.K, bool)
        
        self.action_mask[self.j:] = False
        self.assignment[self.j:] = 0
        self.is_O_allowed[self.j:] = True
        self.A = np.zeros(len(self.D), bool)
        self.A[self.dests[:self.j]] = True
        self.A[self.hub] = True
        self.NA = ~self.A
        self.NA[self.j] = False
        
        self.info = {
            "omitted" : [],
            "episode rewards" : self.episode_reward,
            "quantity accepted" : self.total_capacity - self.remained_capacity,
            "remained capacity" : self.remained_capacity,
            "h" : self.h,
            "j" : self.j,
            "dest" : self.dests[self.j],
        }
        
        
        self.routes = np.zeros((len(self.emissions_KM), self.max_capacity+2), dtype=np.int64)
        # self.routing_data = RoutingData(
        #     self.routes,
        #     self.cost_matrix,
        #     self.distance_matrix,
        #     self.quantities,
        #     self.max_capacity,
        #     self.costs_KM,
        #     self.emissions_KM,
        #     self.Q
        # )
        
        
    def _compute_min_med(self):
        min_knn = np.mean(knn(self.p[self.A]*self.D[self.A, self.j], self.k_min))
        med_knn = np.median(knn(self.p[self.NA]*self.D[self.NA, self.j], self.k_med))
        
        return min_knn, med_knn
    
    def _get_obs(self):
        
        min_knn, med_knn = self._compute_min_med()
        
        obs = np.array([
            self.quantities[self.j]/ self.total_capacity, # the quantity rate asked by the current demand
            self.remained_capacity / self.total_capacity, # the percentage of capacity remained
            (self.H - self.h) / max(1, self.H), # the remaining demands to come
            min_knn/np.max(self.D), # The mean of the k nearest neighbors in admitted dests
            med_knn/np.max(self.D), # The mean of the k nearest neighbors in non activated dests
            max(0, self.info["remained_quota"])/self.Q
            #TODO * Maybe find better observations
        ])
        
        return obs
    
    def reset(self, instance_id = -1):
        
        
        self._init_instance(instance_id)
        
        self.assignment, self.routes, self.info = SA_routing(self)
        
        # print(self.assignment)
        self.remained_capacity -= np.sum(self.quantities[self.assignment.astype(bool)])
        
        obs = self._get_obs()
        
        self.info.update({
            'assignment' : self.assignment,
            "omitted" : [],
            "episode rewards" : self.episode_reward,
            "quantity accepted" : self.total_capacity - self.remained_capacity,
            "remained capacity" : self.remained_capacity,
            "h" : self.h,
            "j" : self.j,
            "dest" : self.dests[self.j],
        })
        
        return obs, self.info
    
    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        
        if self.h >= self.H-1:
            print("The episode is done :")
            print(self.info)
            return
            
        self.h += 1
        assert isinstance(action, int)
        
        self.NA[self.j] = False
        
        if action:
            
            if self.re_optimization:
                self.assignment, self.routes, self.info = SA_routing(self)
            else:
                self.assignment, self.routes, self.info = insertion(self)

            if not self.assignment[self.j]:
                action = 0
                
        if action:
            self.action_mask[self.j] = True
            self.is_O_allowed[self.j] = False
            self.A[self.j] = True
            r = self.quantities[self.j]
            self.episode_reward += r
            self.remained_capacity -= r

        else:
            r = 0
            self.omitted.append(self.j)
            
        self.j += 1
        self.info.update({
            'assignment' : self.assignment,
            "episode rewards" : self.episode_reward,
            "quantity accepted" : self.total_capacity - self.remained_capacity,
            "remained capacity" : self.remained_capacity,
            "omitted" : self.dests[self.omitted],
            "h" : self.h,
            "j" : self.j,
            "dest" : self.dests[self.j],
        })
        
        obs = self._get_obs()
        
        trunc = self.h >= self.H-1
        done = trunc or self.info["remained_quota"] <= 1e-4 or self.remained_capacity <= 0
        # info['r'] = np.clip((normalizer_const + info['r'])/normalizer_const, 0, 1)
        
        return obs, r, done, trunc, self.info
    
    def sample(self, H):
        
        env = deepcopy(self)
        p = env.p.copy()
        p[env.dests[:self.j]] = 0
        p[env.hub] = 0
        p /= p.sum()
        
        H = min(H, env.H - env.h - 1)
        
        future_dests = np.random.choice(len(p), H, False, p)
        env.dests[self.j+1 : self.j+H+1] = future_dests
        
        l = [env.hub] + list(env.dests)
        env.mask = np.ix_(l, l)
        env.distance_matrix = env.D[env.mask]
        env.cost_matrix = np.array([
            (env.costs_KM[v] + env.CO2_penalty*env.emissions_KM[v])*env.distance_matrix
            for v in range(len(env.costs_KM))
        ])
        # TODO * implement quantity sampling
        
        env.action_mask[:self.j+H] = True
        return SA_routing(env)
    
    def offline_solution(self, *args, **kwargs):
        env = deepcopy(self)
        env.action_mask[:] = True
        return SA_routing(env, *args, **kwargs)
        
    
    def render(self):
        G = nx.DiGraph()
        G.add_nodes_from(list(range(self.j)))
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
                #     #TODO *
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
                         'pos' : (self.coordx[self.hub], self.coordy[self.hub]),
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

        plt.ylim(min(self.coordy[self.dests]) - 1, max(self.coordy[self.dests])+1)
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