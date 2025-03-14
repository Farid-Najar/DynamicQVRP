from copy import deepcopy
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
from numba import njit

from methods.OR.routing import SA_routing, SA_routing2, insertion
from utils.generate_scenarios import create_random_scenarios
# from methods import VA_SA

from typing import Any, Dict, Optional
import gymnasium as gym

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
    k = min(len(a)-1, k)
    idx = np.argpartition(a, k)
    # print('knn', idx[1:k+2])
    # print('a knn', a)
    # print('a knn', a[idx[0:k+1]].mean())
    return idx[:k+1]

def load_data(cluster_scenario = False):
    
    if cluster_scenario:
        D = np.load('data/clusters/D_cluster.npy')
        coordx = np.load('data/clusters/coordx_cluster.npy')
        coordy = np.load('data/clusters/coordy_cluster.npy')
        return D, coordx, coordy, np.ones(D.shape[0])/D.shape[0]
    
    coordx = np.load('data/coordsX.npy')
    coordy = np.load('data/coordsY.npy')
    D = np.load('data/distance_matrix.npy')
    probs = np.load('data/prob_dests.npy')
    
    return D, coordx, coordy, probs

@njit
def calculate_marginal_emission(routes, E, d):
    """Calculate the minimal marginal emission cost of inserting destination d into each route.
    
    For each route, this function finds the optimal insertion point that minimizes
    the additional emission cost using the triangular difference approach.
    
    Parameters
    ----------
    routes : np.ndarray
        Array of routes, where each row represents a vehicle's route
    E : np.ndarray
        Emission matrices, where E[v] is the emission matrix for vehicle v
    d : int
        Destination to insert
        
    Returns
    -------
    np.ndarray
        Array of minimal marginal emission costs for each route
    """
    num_vehicles = len(routes)
    marginal_costs = np.full(num_vehicles, np.amax(E))
    
    for v in range(num_vehicles):
        route = routes[v]
        min_additional_emission = np.inf
            
        # Try inserting d at each possible position using triangular difference
        for j in range(len(route)-1):
            prev_node = route[j]
            next_node = route[j+1]
            
            if j and prev_node + next_node == 0:
                break
            
            # Calculate triangular difference: d(i,j) + d(j,k) - d(i,k)
            # where i=prev_node, j=d, k=next_node
            additional_emission = \
                E[v][prev_node, d] + E[v][d, next_node] - E[v][prev_node, next_node]
            
            min_additional_emission = min(min_additional_emission, additional_emission)
        
        marginal_costs[v] = min_additional_emission
    
    return marginal_costs


class DynamicQVRPEnv(gym.Env):
    """
    DynamicQVRPEnv is a custom environment for the Dynamic Quadratic Vehicle Routing Problem (QVRP).
    
    This environment simulates a dynamic routing problem where vehicles must be routed to meet demands
    at various destinations while minimizing costs and emissions. The environment supports various 
    configurations and allows for re-optimization during the routing process.

    Attributes
    ----------
    instance : int
        The current instance of the environment.
    D : np.ndarray
        Distance matrix.
    coordx : np.ndarray
        X coordinates of the destinations.
    coordy : np.ndarray
        Y coordinates of the destinations.
    p : np.ndarray
        Probability of destinations.
    all_dests : np.ndarray
        All possible destinations in the scenario set.
    qs : np.ndarray
        Quantities for each destination.
    max_capacity : int
        Maximum capacity of a vehicle.
    total_capacity : int
        Total capacity of all vehicles.
    H : int
        Number of demands to be met.
    K : int
        Horizon length.
    q : np.ndarray
        Quantities for each destination.
    omission_cost : float
        Cost of omitting a destination.
    CO2_penalty : float
        Penalty for CO2 emissions.
    costs_KM : list
        Costs per kilometer for each vehicle.
    emissions_KM : list
        Emissions per kilometer for each vehicle.
    num_actions : int
        Number of possible actions.
    re_optimization : bool
        Flag to indicate if re-optimization is allowed.
    observation_space : gym.spaces.Box
        Observation space for the environment.
    action_space : gym.spaces.Discrete
        Action space for the environment.
    allow_initial_omission : bool
        Flag to allow initial omission of destinations.
    remained_capacity : int
        Remaining capacity of the vehicles.
    h : int
        Current step in the horizon.
    j : int
        Current destination index.
    assignment : np.ndarray
        Assignment of destinations to vehicles.
    action_mask : np.ndarray
        Mask for valid actions.
    is_O_allowed : np.ndarray
        Mask for allowed omissions.
    A : np.ndarray
        Array indicating activated destinations.
    NA : np.ndarray
        Array indicating non-activated destinations.
    info : dict
        Dictionary containing information about the current state.
    routes : np.ndarray
        Routes for each vehicle.
    omitted : list
        List of omitted destinations.
    episode_reward : float
        Total reward for the current episode.

    Methods
    -------
    _init_instance(instance_id)
        Initialize a new instance of the environment.
    _compute_min_med()
        Compute the minimum and median distances for the current state.
    _get_obs()
        Get the current observation.
    reset(instance_id=-1, *args, **kwargs)
        Reset the environment to a new state.
    step(action)
        Take a step in the environment.
    sample(H)
        Sample a future state of the environment.
    offline_solution(*args, **kwargs)
        Compute an offline solution for the environment.
    render(size=100, show_node_num=False)
        Render the current state of the environment.
    """
    
    def __init__(self, 
                #  env : AssignmentEnv = None,
                 horizon = 50,
                 Q = 50,
                 DoD = .5,
                 vehicle_capacity = 15, # We assume it homogeneous for all vehicles
                 retain_rate = 0.,
                 use_dataset = True,
                 re_optimization  = False,
                 costs_KM = [1], 
                 emissions_KM = [.3], 
                 CO2_penalty = 10_000,
                 k_min : int = 3,
                 k_med : int = 7,
                 n_scenarios = None,
                 hub = 0,
                 test = False,
                 allow_initial_omission = True,
                 unknown_p = False,
                 uniforme_p_test = False,
                 noised_p = False,
                 different_quantities = False,
                 vehicle_assignment = False,
                 cluster_scenario = False,
                 static_as_dynamic = False,
                 noise_horizon = 0., # Represents the percentage of the noise in horizon. in [0, 1]
                 seed = 1917,
        ):
        
        K = horizon
        self.instance = -1
        self.D, self.coordx, self.coordy, self.p = load_data(cluster_scenario)
        
        self.emissions_KM = emissions_KM
        self.E = np.array([
            self.emissions_KM[v]*self.D
            for v in range(len(self.emissions_KM))
        ])
        
        np.random.seed(seed)
        
        use_dataset = test or use_dataset
        
        if use_dataset and not cluster_scenario:
            retain_comment = f"_retain{retain_rate}" if retain_rate else ""
            scenario_comment = f"_{n_scenarios}" if n_scenarios is not None else ""
            noise_comment = f"noised_" if noised_p else ""
            uniforme = f"_uniforme" if uniforme_p_test else ""
            # with open(f'data/game_K{K}{retain_comment}.pkl', 'rb') as f:
            #     g = pickle.load(f)
            # routes = np.load(f'data/routes_K{K}{retain_comment}.npy')
            if test:
                self.all_dests = np.load(f'data/{noise_comment}destinations_K{K}_100{uniforme}_test.npy').astype(int)
            else:
                self.all_dests = np.load(f'data/destinations_K{K}{retain_comment}{scenario_comment}{uniforme}.npy').astype(int)
                
        else:
            if test and cluster_scenario:
                self.all_dests = np.load(f'data/clusters/destinations_K{K}_101_test.npy').astype(int)
               
            else: 
                self.all_dests = create_random_scenarios(
                    n_scenarios = n_scenarios if n_scenarios is not None else 500,
                    d = K,
                    hub = hub,
                    save = False,
                    p = self.p
                )#np.random.choice(len(self.D), n_scenarios, True)+1
                # raise("not implemented yet")
        
        if different_quantities:
            qs = np.random.randint(1, vehicle_capacity//4, (len(self.all_dests), K))
            #np.load(f'data/quantities_K{K}_retain1.0.npy')
        else:
            qs = np.ones((len(self.all_dests), K))
            
        self.max_capacity = vehicle_capacity
        self.total_capacity = vehicle_capacity*len(emissions_KM)
        DoD = 1-(self.total_capacity-DoD*self.total_capacity)/K
        self.T = int(DoD*K) # ou = self.H
        self.H = K
        
        self.noise_horizon = noise_horizon
        
        if unknown_p or static_as_dynamic:
            self.p[:] = 1.
            
        self.static_as_dynamic = static_as_dynamic
        
        self.qs = qs
        # self.omission_cost = self._env.omission_cost
        # self.CO2_penalty = self._env.CO2_penalty
        self.Q = Q
        self.hub = hub
        self.k_min = k_min
        self.k_med = k_med
        self.costs_KM = costs_KM
        # self.is_0_allowed = is_0_allowed
        self.num_actions = len(self.emissions_KM) + 1 #if is_0_allowed else len(self.emissions_KM)
        self.re_optimization = re_optimization
        
        self.CO2_penalty = CO2_penalty
        self.omission_cost = (2*np.max(self.D) +1)*np.max(self.costs_KM)
        
        # * Change if obs change
        # self.observation_space = gym.spaces.Box(0, 1, (5+len(emissions_KM),), np.float64) 
        dim_obs = 4 + 2*len(emissions_KM)# if not vehicle_assignment else 5 + len(emissions_KM) + len(self.emissions_KM)
        self.observation_space = gym.spaces.Box(0, 1, (dim_obs,), np.float64)
        # self.observation_space = gym.spaces.Box(0, 1, (6,), np.float_)
        
        # * change if actions change
        self.vehicle_assignment = vehicle_assignment
        dim_actions = 2 if not vehicle_assignment else len(self.emissions_KM) + 1 
        self.action_space = gym.spaces.Discrete(dim_actions)
        # self.action_space = gym.spaces.Discrete(2)
        
        self.allow_initial_omission = allow_initial_omission
        
        
    def _init_instance(self, instance_id):
        # self._env.reset(calculate_routes=False)
        
        self.h = 0 # ou = self.H - int(DoD*K)
        if instance_id < 0:
            self.instance = (self.instance+1)%len(self.all_dests)
        else:
            self.instance = instance_id
            
        np.random.seed(None)
        self.noised_H = self.H + int(np.random.normal(0, self.noise_horizon*self.H))
        
        self.dests = self.all_dests[self.instance]
        self.quantities = self.qs[self.instance]
        
        self.remained_capacity = self.total_capacity
        
        l = [self.hub] + list(self.dests)
        self.mask = np.ix_(l, l)
        self.distance_matrix = self.D[self.mask]
        # self.emission_matrices = self.E[self.mask]
        self.cost_matrix = np.array([
            (self.costs_KM[v] + self.CO2_penalty*self.emissions_KM[v])*self.distance_matrix
            for v in range(len(self.emissions_KM))
        ])
        self.omitted = []
        
        self.episode_reward = 0
        
        # self.dests = self.all_dests[self.instance]
        self.t = self.H - self.T
        self.assignment = np.ones(self.H, int)
        
        self.action_mask = np.ones(self.H, bool)
        self.is_O_allowed = np.ones(self.H, bool)
        
        self.action_mask[self.t:] = False
        if not self.allow_initial_omission:
            self.is_O_allowed[:self.t] = False
            
        self.assignment[self.t:] = 0
        self.A = np.zeros(len(self.D), bool)
        self.A[self.dests[:self.t]] = True
        self.A[self.hub] = True
        if self.static_as_dynamic:
            self.NA = np.zeros(len(self.D), bool)
            self.NA[self.dests[self.t+1:]] = True
        else:
            self.NA = ~self.A
            self.NA[self.dests[self.t]] = False
        
        self.info = {
            "omitted" : [],
            "remained_quota" : self.Q,
            "episode rewards" : self.episode_reward,
            "quantity accepted" : self.total_capacity - self.remained_capacity,
            "remained capacity" : self.remained_capacity,
            "h" : self.h,
            "t" : self.t,
            "dest" : self.dests[self.t],
        }
        
        
        self.routes = np.zeros((len(self.emissions_KM), self.max_capacity+2), dtype=np.int64)
        
        if self.t:
            self.assignment, self.routes, self.info = SA_routing2(self)
        
        self.omitted += list(np.where(self.assignment[:self.t]==0)[0])
        
        self.is_O_allowed[:self.t] = False
        
        if "remained_quota" not in self.info.keys():
            raise (Exception("The Quota Q might be too low."))
        
        self.action_mask[self.t] = True
        
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
        if self.static_as_dynamic:
            p = np.ones_like(self.p[self.NA])
        else:
            p = self.p[self.NA].copy()
            # p[~self.NA] = 0
            p /= p.sum()
        
        # * The mean of the k nearest neighbors in admitted dests for every vehicle
        masks = [
            np.concatenate([[self.hub], self.dests[np.where(self.assignment == v)[0]]])
            for v in range(1, len(self.costs_KM)+1)
        ]
        # min_knn = np.median([
        #     np.mean(knn(
        #         self.D[mask, self.dests[self.t]], self.k_min
        #     ))
        #     for mask in masks if len(mask)
        # ])
        
        D_V = [
            self.D[mask, self.dests[self.t]]
            for mask in masks if len(mask)
        ]

        min_knn = np.array([
            self.emissions_KM[v]*np.mean(
                D_V[v][knn(
                    D_V[v], self.k_min
                    )]
            )
            for v in range(len(D_V))
        ])
        # print(self.D[masks[0], self.dests[self.t]])
        # print(self.D[masks[1], self.dests[self.t]])
        # print(min_knn)
    
        # D_A = self.D[self.A, self.dests[self.t]]
        # min_knn = np.mean(D_A[knn(D_A, self.k_min)])
        
        # * The mean of the k nearest neighbors in non admitted dests
        D_NA = self.D[self.NA, self.dests[self.t]]
        
        idx_NA = knn(D_NA, self.k_med)
        med_knn = np.median(p[idx_NA]*D_NA[idx_NA])
        # med_knn = np.median(knn(self.D[self.NA, self.dests[self.t]]/(p[self.NA] + 1e-8), self.k_med))
        
        
        return min_knn, med_knn
    
    def _get_obs(self):
        
        min_knn, med_knn = self._compute_min_med()
        cap = np.full(len(self.costs_KM), 1.)
        cap[:self.assignment.max()] -= (
            np.bincount(self.assignment)[1:self.assignment.max()+1]
        )/self.max_capacity
        
        if self.noise_horizon:
            remaining_demands = (self.noised_H - self.t)/self.noised_H
        else:
            remaining_demands = (self.T - self.h) / self.H
        
        obs = np.array([
            self.quantities[self.t]/ self.total_capacity, # the quantity rate asked by the current demand
            # self.remained_capacity / self.total_capacity, # the percentage of capacity remained
            *cap, # the percentage of capacity remained for each vehicle, dim = len(self.emissions_KM)
            remaining_demands,#, # the remaining demands to come
            *min_knn/(np.amax(self.D)*max(self.emissions_KM)), # The mean emissions of the k nearest neighbors in admitted dests, dim = len(self.emissions_KM)
            med_knn/(np.amax(self.D)), # The mean of the k nearest neighbors in non activated dests
            # med_knn/(np.max(self.D)/1e-8), # The mean of the k nearest neighbors in non activated dests
            max(0, self.info["remained_quota"])/self.Q, # The remaining quota
            # * the emissions have been removed from the observation
            # Instead, it has directly been integrated into the distances
            # See the _compute_min_med method
            # *self.emissions_KM, # emission of each vehicle, dim = len(self.emissions_KM)
            # * TODO : Maybe find better observations
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
            "j" : self.t,
            "quantity demanded" : self.quantities[self.t],
            "dest" : self.dests[self.t],
        })
        
        return obs, self.info
    
    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        
        if (self.h >= self.T-1 or 
            self.remained_capacity <= 0 or
            self.info["remained_quota"] <= 1e-4 
        ):
            print("The episode is done :")
            print(self.info)
            return -1, 0, True, True, self.info
            
        self.h += 1
        assert isinstance(action, (int, np.int_)), f"type : {type(action)}, {action}"
        
        current_dest = self.dests[self.t]
        self.NA[current_dest] = False
        
        if action:
            # action = action if self.vehicle_assignment else None
                
            if self.re_optimization:
                self.assignment, self.routes, self.info = insertion(self)
                self.assignment, self.routes, self.info = SA_routing2(self)
                
            elif self.vehicle_assignment:
                self.assignment, self.routes, self.info = insertion(self, action, run_sa=True)
                # self.assignment, self.routes, self.info = SA_routing2(self, multiple_tsp=True)
            else:
                self.assignment, self.routes, self.info = insertion(self)
                
            if not self.assignment[self.t]:
                action = 0
                
        if action:
            self.is_O_allowed[self.t] = False
            self.A[current_dest] = True
            r = self.quantities[self.t]
            self.episode_reward += r
            self.remained_capacity -= r

        else:
            self.action_mask[self.t] = False
            r = 0
            self.omitted.append(self.t)
            
        self.t += 1
        self.action_mask[self.t] = True
        self.info.update({
            'assignment' : self.assignment,
            "episode rewards" : self.episode_reward,
            "quantity accepted" : self.total_capacity - self.remained_capacity,
            "remained capacity" : self.remained_capacity,
            "omitted" : self.dests[self.omitted],
            "h" : self.h,
            "t" : self.t,
            "quantity demanded" : self.quantities[self.t],
            "dest" : current_dest,
        })
        
        obs = self._get_obs()
        
        trunc = self.h >= self.T-1
        done = bool(trunc or self.info["remained_quota"] <= 1e-4 or self.remained_capacity <= 0)
        # info['r'] = np.clip((normalizer_const + info['r'])/normalizer_const, 0, 1)
        
        return obs, r, done, trunc, self.info
    
    def sample(self, H, SA_configs):
        
        env = deepcopy(self)
        p = env.p.copy()
        p[env.dests[:self.t]] = 0
        p[env.hub] = 0
        
        H = min(H, env.H - env.h - 1)
        
        if len(self.cost_matrix) > 1:
            env.action_mask[:self.t + H+1] = True
        else:
            env.action_mask = env.is_O_allowed.copy()
            env.action_mask[H+1:] = False
            
        p /= p.sum()
        
        
        future_dests = np.random.choice(len(p), H, False, p)
        env.dests[self.t+1 : self.t+H+1] = future_dests
        
        l = [env.hub] + list(env.dests)
        env.mask = np.ix_(l, l)
        env.distance_matrix = env.D[env.mask]
        env.cost_matrix = np.array([
            (env.costs_KM[v] + env.CO2_penalty*env.emissions_KM[v])*env.distance_matrix
            for v in range(len(env.costs_KM))
        ])
        # TODO : implement quantity sampling
        
        # env.action_mask[:self.t+H] = True
        return SA_routing2(env, offline_mode=True, **SA_configs)
    
    def offline_solution(self, *args, **kwargs):
        env = deepcopy(self)
        if len(self.cost_matrix) > 1:
            env.action_mask[:] = True
        else:
            env.action_mask = env.is_O_allowed.copy()
        env.H = 0
        return SA_routing2(env, offline_mode=True, *args, **kwargs)
        
    
    def render(self,
               size = 100, show_node_num =False, display_current_node = True,
               display_unactivated = True, display_dests = False,
               color_bar_label = None,
               ):
        # print(self.assignment)
        G = nx.DiGraph()
        G.add_nodes_from(list(range(self.t+1)))
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
                        self.emissions_KM[m]*self.distance_matrix[
                            int(self.routes[m, j-1]), int(self.routes[m, j])
                        ]
                    )
                )
                # vehicle_edges[m].append((int(route[m, j]), int(route[m, j+2])))
                j+=1
            if int(self.routes[m, j-1]) != int(self.routes[m, j]):
                edges.append(
                        (
                            int(self.routes[m, j-1]),
                            int(self.routes[m, j]),
                            self.emissions_KM[m]*self.distance_matrix[
                                int(self.routes[m, j-1]), int(self.routes[m, j])
                            ]
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
        G_ncolors[0] = 'black'

        _, ax = plt.subplots(figsize=(12, 7))
        weights = list(nx.get_edge_attributes(G,'weight').values())
        
        if self.static_as_dynamic:
            p = np.zeros_like(self.p)
            p[self.NA] = .01
        else:
            p = self.p.copy()
            p[~self.NA] = 0
            p /= p.sum()
        
        # ax.scatter(self.coordx[self.dests], self.coordy[self.dests], color='lightgray', s = .6*size, label='Unactivated')
        if display_unactivated:
            ax.scatter(
                self.coordx[self.NA], self.coordy[self.NA], 
                color='lightgray', 
                s = 100*p[self.NA]*size, 
                label='Unactivated'
            )
        
        if display_dests:
            ax.scatter(
                self.coordx[self.dests], self.coordy[self.dests], 
                color='gray', 
                s = size, 
                label='Destinations'
            )
            
        # print(self.coordx[self.dests[self.t]], self.coordy[self.dests[self.t]])
        if display_current_node:
            ax.scatter(
                self.coordx[self.dests[self.t]], self.coordy[self.dests[self.t]], 
                s = size*self.quantities[self.t], 
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
        ax.scatter([0],[0],color='black', s = size, marker='s', label='Hub')
        ax.scatter([0],[0],color='white', s = size, marker='s')

        # reverse the order
        plt.draw()
        lgnd = plt.legend(bbox_to_anchor=(1.4, 1.0), loc='upper right')
        # lgnd = plt.legend(loc="lower left", scatterpoints=1, fontsize=10)
        for handle in lgnd.legend_handles:
            handle.set_sizes([50])
        mesh = ax.pcolormesh(([], []), cmap = plt.cm.jet)
        try:
            mesh.set_clim(np.min(weights),np.max(weights))
        except:
            pass
        # Visualizing colorbar part -start
        cbar = plt.colorbar(
            mesh,
            ax=ax, 
            label = color_bar_label if color_bar_label is not None else 'emissions (in kg CO2)',
        )
        cbar.formatter.set_powerlimits((0, 0))
        # to get 10^3 instead of 1e3
        cbar.formatter.set_useMathText(True)

        # plt.colorbar()
        # plt.style.use("dark_background")
        # plt.legend()
        plt.show()

        return nx.to_latex(
            G,
            nx.get_node_attributes(G,'pos'), 
            node_options=dict(zip(range(len(G_ncolors)), G_ncolors))
        )
    
    
class StaticWrapper:
    # It recreates the conditions of the static case when the demands are known
    # TODO
    def __init__(self, env : DynamicQVRPEnv):
        self._env = env