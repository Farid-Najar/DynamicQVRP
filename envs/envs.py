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
    """It computes the k nearest neighbors (k mins of array a)
    If k > len(a), it takes k = len(a)

    Parameters
    ----------
    a : np.ndarray
        The array
    k : int
        The number of mins

    Returns
    -------
    np.ndarray
        k mins of array a
    """
    idx = np.argpartition(a, min(len(a)-1, k))
    return idx[:k]

def load_data():
    coordx = np.load('data/coordsX.npy')
    coordy = np.load('data/coordsY.npy')
    D = np.load('data/distance_matrix.npy')
    probs = np.load('data/prob_dests.npy')
    
    return D, coordx, coordy, probs


class DynamicQVRPEnv(gym.Env):
    
    def __init__(self, 
                #  env : AssignmentEnv = None,
                 horizon = 50,
                 Q = 50,
                 DoD = 0.,
                 vehicle_capacity = 15, # We assume it homogeneous for all vehicles
                 retain_rate = 0.,
                 use_dataset = True,
                 re_optimization  = False,
                 costs_KM = [1], 
                 emissions_KM = [.3], 
                 CO2_penalty = 10_000,
                 k_min : int = 3,
                 k_med : int = 5,
                 n_scenarios = None,
                 hub = 0,
                 test = False,
                 allow_initial_omission = True,
                 unknown_p = False,
                 different_quantities = False,
        ):
        
        K = horizon
        self.instance = -1
        self.D, self.coordx, self.coordy, self.p = load_data()
        
        if use_dataset:
            retain_comment = f"_retain{retain_rate}" if retain_rate else ""
            scenario_comment = f"_{n_scenarios}" if n_scenarios is not None else ""
            # with open(f'data/game_K{K}{retain_comment}.pkl', 'rb') as f:
            #     g = pickle.load(f)
            # routes = np.load(f'data/routes_K{K}{retain_comment}.npy')
            if test:
                self.all_dests = np.load(f'data/destinations_K{K}{retain_comment}{scenario_comment}_test.npy').astype(int)
            else:
                self.all_dests = np.load(f'data/destinations_K{K}{retain_comment}{scenario_comment}.npy').astype(int)
                
        else:
            #TODO *** Implement the generation of the destinations
            self.all_dests = np.random.choice(len(self.D), n_scenarios, True)+1
            raise("not implemented yet")
        
        if different_quantities:
            qs = np.random.randint(1, vehicle_capacity//4, (len(self.all_dests), K))
            #np.load(f'data/quantities_K{K}_retain1.0.npy')
        else:
            qs = np.ones((len(self.all_dests), K))
            
        self.max_capacity = vehicle_capacity
        self.total_capacity = vehicle_capacity*len(costs_KM)
        DoD = 1-(self.total_capacity-DoD*self.total_capacity)/K
        self.H = int(DoD*K) # ou = self.K
        self.K = K
        
        if unknown_p:
            self.p[:] = 1.
        
        self.qs = qs
        # self.omission_cost = self._env.omission_cost
        # self.CO2_penalty = self._env.CO2_penalty
        self.Q = Q
        self.hub = hub
        self.k_min = k_min
        self.k_med = k_med
        self.emissions_KM = emissions_KM
        self.costs_KM = costs_KM
        # self.is_0_allowed = is_0_allowed
        self.num_actions = len(self.emissions_KM) + 1 #if is_0_allowed else len(self.emissions_KM)
        self.re_optimization = re_optimization
        
        self.CO2_penalty = CO2_penalty
        self.omission_cost = (2*np.max(self.D) +1)*np.max(self.costs_KM)
        
        self.observation_space = gym.spaces.Box(0, 1, (6,), np.float64) #TODO * Change if obs change
        self.action_space = gym.spaces.Discrete(2)
        
        self.allow_initial_omission = allow_initial_omission
        
        
    def _init_instance(self, instance_id):
        # self._env.reset(calculate_routes=False)
        
        self.h = 0 # ou = self.K - int(DoD*K)
        if instance_id < 0:
            self.instance = (self.instance+1)%len(self.all_dests)
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
        self.is_O_allowed = np.ones(self.K, bool)
        
        self.action_mask[self.j:] = False
        if not self.allow_initial_omission:
            self.is_O_allowed[:self.j] = False
            
        self.assignment[self.j:] = 0
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
        
        self.assignment, self.routes, self.info = SA_routing(self)
        
        self.omitted += list(np.where(self.assignment[:self.j]==0)[0])
        
        self.is_O_allowed[:self.j] = False
        
        if "remained_quota" not in self.info.keys():
            raise (Exception("The Quota Q might be too low."))
        
        self.action_mask[self.j] = True
        
        self.remained_capacity -= np.sum(self.quantities[self.assignment.astype(bool)])
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
        p = self.p[self.NA].copy()
        # p[~self.NA] = 0
        p /= p.sum()
        
        # masks = [
        #     np.concatenate([[self.hub], self.dests[np.where(self.assignment == v)[0]]])
        #     for v in range(1, len(self.costs_KM)+1)
        # ]
        # min_knn = np.median([
        #     np.mean(knn(
        #         self.D[mask, self.dests[self.j]], self.k_min
        #     ))
        #     for mask in masks if len(mask)
        # ])
        D_A = self.D[self.A, self.dests[self.j]]
        D_NA = self.D[self.NA, self.dests[self.j]]
        
        min_knn = np.mean(D_A[knn(D_A, self.k_min)])
        idx_NA = knn(D_NA, self.k_med)
        med_knn = np.median(p[idx_NA]*D_NA[idx_NA])
        # med_knn = np.median(knn(self.D[self.NA, self.dests[self.j]]/(p[self.NA] + 1e-8), self.k_med))
        
        
        return min_knn, med_knn
    
    def _get_obs(self):
        
        min_knn, med_knn = self._compute_min_med()
        
        obs = np.array([
            self.quantities[self.j]/ self.total_capacity, # the quantity rate asked by the current demand
            self.remained_capacity / self.total_capacity, # the percentage of capacity remained
            (self.H - self.h) / max(1, self.H), # the remaining demands to come
            min_knn/np.max(self.D), # The mean of the k nearest neighbors in admitted dests
            med_knn/(np.max(self.D)), # The mean of the k nearest neighbors in non activated dests
            # med_knn/(np.max(self.D)/1e-8), # The mean of the k nearest neighbors in non activated dests
            max(0, self.info["remained_quota"])/self.Q
            #TODO * Maybe find better observations
        ])
        
        return obs
    
    def reset(self, instance_id = -1, *args, **kwargs):
        
        
        self._init_instance(instance_id)
        
        obs = self._get_obs()
        
        self.info.update({
            'assignment' : self.assignment,
            "omitted" : [],
            "episode rewards" : self.episode_reward,
            "quantity accepted" : self.total_capacity - self.remained_capacity,
            "remained capacity" : self.remained_capacity,
            "h" : self.h,
            "j" : self.j,
            "quantity demanded" : self.quantities[self.j],
            "dest" : self.dests[self.j],
        })
        
        return obs, self.info
    
    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        
        if self.h >= self.H-1 or self.remained_capacity <= 0:
            print("The episode is done :")
            print(self.info)
            return -1, 0, True, True, self.info
            
        self.h += 1
        assert isinstance(action, (int, np.int_)), f"type : {type(action)}, {action}"
        
        self.NA[self.j] = False
        
        if action:
            
            if self.re_optimization:
                self.assignment, self.routes, self.info = insertion(self)
                self.assignment, self.routes, self.info = SA_routing(self)
            else:
                self.assignment, self.routes, self.info = insertion(self)

            if not self.assignment[self.j]:
                action = 0
                
        if action:
            self.is_O_allowed[self.j] = False
            self.A[self.j] = True
            r = self.quantities[self.j]
            self.episode_reward += r
            self.remained_capacity -= r

        else:
            self.action_mask[self.j] = False
            r = 0
            self.omitted.append(self.j)
            
        self.j += 1
        self.action_mask[self.j] = True
        self.info.update({
            'assignment' : self.assignment,
            "episode rewards" : self.episode_reward,
            "quantity accepted" : self.total_capacity - self.remained_capacity,
            "remained capacity" : self.remained_capacity,
            "omitted" : self.dests[self.omitted],
            "h" : self.h,
            "j" : self.j,
            "quantity demanded" : self.quantities[self.j],
            "dest" : self.dests[self.j],
        })
        
        obs = self._get_obs()
        
        trunc = self.h >= self.H-1
        done = bool(trunc or self.info["remained_quota"] <= 1e-4 or self.remained_capacity <= 0)
        # info['r'] = np.clip((normalizer_const + info['r'])/normalizer_const, 0, 1)
        
        return obs, r, done, trunc, self.info
    
    def sample(self, H):
        
        env = deepcopy(self)
        p = env.p.copy()
        p[env.dests[:self.j]] = 0
        p[env.hub] = 0
        
        if len(self.cost_matrix) > 1:
            env.action_mask[:] = True
        else:
            env.action_mask = env.is_O_allowed.copy()
            
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
        if len(self.cost_matrix) > 1:
            env.action_mask[:] = True
        else:
            env.action_mask = env.is_O_allowed.copy()
        env.H = 0
        return SA_routing(env, *args, **kwargs)
        
    
    def render(self, size = 100, show_node_num =False):
        # print(self.assignment)
        G = nx.DiGraph()
        G.add_nodes_from(list(range(self.j+1)))
        # Go = nx.DiGraph()
        # Go.add_nodes_from(self.omitted)
        node_attrs = dict()
        edges = []
        # vehicle_edges = []
        # print(self.routes.shape)
        for o in self.omitted:
            node_attrs[o+1] = {
                         'vehicle' : 0, 
                        #  'x' : [self.dests[int(self.routes[m, j])], m, gained_on_substitution], 
                         'pos' : (self.coordx[self.dests[o]], self.coordy[self.dests[o]]),
                         'q' : size*self.quantities[o],
            }
        
        # Go_attrs = deepcopy(node_attrs)
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
                         'q' : size*self.quantities[self.routes[m, j]-1],
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
                         'q' : size*2
                }
        G.add_weighted_edges_from(edges)
        # print(G.nodes)
        nx.set_node_attributes(G, node_attrs)
        # nx.set_node_attributes(Go, Go_attrs)
        
        colors = []#'#1f78b4' for _ in range(len(G.nodes))]
        # colors.append('lightgray')
        colors.append('red')
        colors.append('lightgreen')
        if len(self.emissions_KM) > 3:
            colors.append('lightblue')
        colors.append('yellow')
        colors.append('lightcoral')#'red')
        G_ncolors = [colors[m] for m in nx.get_node_attributes(G,'vehicle').values()]
        G_ncolors[0] = 'gray'

        _, ax = plt.subplots(figsize=(12, 7))
        weights = list(nx.get_edge_attributes(G,'weight').values())
        
        p = self.p.copy()
        p[~self.NA] = 0
        p /= p.sum()
        
        # ax.scatter(self.coordx[self.dests], self.coordy[self.dests], color='lightgray', s = .6*size, label='Unactivated')
        ax.scatter(self.coordx[self.NA], self.coordy[self.NA], color='lightgray', s = 100*p[self.NA]*size, label='Unactivated')
        # print(self.coordx[self.dests[self.j]], self.coordy[self.dests[self.j]])
        ax.scatter(
            self.coordx[self.dests[self.j]], self.coordy[self.dests[self.j]], 
            s = size*self.quantities[self.j], 
            color='blue', 
            label='Current demand'
        )
        nx.draw_networkx(G, 
                         pos = nx.get_node_attributes(G,'pos'),  
                         ax=ax, 
                         font_size=5, 
                         with_labels=show_node_num,
                         node_size=list(nx.get_node_attributes(G,'q').values()), 
                         node_color=G_ncolors,
                         edge_color = weights,
                         edge_cmap=plt.cm.jet,
                         node_shape='s',
                         arrows=True
        )
        
        # ax.scatter(self.coordx[self.dests[self.omitted]], self.coordy[self.dests[self.omitted]], edgecolors='red', marker='o', s=size*2, facecolors='none')
        # nx.draw_networkx(Go, 
        #                  pos = nx.get_node_attributes(Go,'pos'),  
        #                  ax=ax, 
        #                  font_size=5, 
        #                  node_size=list(nx.get_node_attributes(Go,'q').values()), 
        #                  node_color='red',
        #                  node_shape='x',
        # )

        plt.ylim(min(self.coordy[self.dests]) - 1, max(self.coordy[self.dests])+1)
        # handles, labels = ax.get_legend_handles_labels()
        # labels = list(range(len(colors)))
        ax.scatter([0],[0],color=colors[0],label=f'Omitted', s = size, marker='s')
        for i in range(1, len(self.routes)+1):
            ax.scatter([0],[0],color=colors[i],label=f'Vehicle {i}', s = size, marker='s')
        ax.scatter([0],[0],color='white', s = size, marker='s')

        # reverse the order
        plt.draw()
        lgnd = plt.legend(bbox_to_anchor=(1.4, 1.0), loc='upper right')
        # lgnd = plt.legend(loc="lower left", scatterpoints=1, fontsize=10)
        for handle in lgnd.legend_handles:
            handle.set_sizes([50])
        mesh = ax.pcolormesh(([], []), cmap = plt.cm.jet)
        if self.j or self.h:
            mesh.set_clim(np.min(weights),np.max(weights))
        # Visualizing colorbar part -start
        cbar = plt.colorbar(mesh,ax=ax)
        cbar.formatter.set_powerlimits((0, 0))
        # to get 10^3 instead of 1e3
        cbar.formatter.set_useMathText(True)

        # plt.colorbar()
        # plt.style.use("dark_background")
        # plt.legend()
        plt.show()

        return nx.to_latex(G, nx.get_node_attributes(G,'pos'), node_options=dict(zip(range(len(G_ncolors)), G_ncolors)))