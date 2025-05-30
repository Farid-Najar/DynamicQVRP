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

@njit(parallel = True)
def generate_D(n, grid_size):
    """Generate a distance matrix D for n locations with uniform random coordinates.
    Parameters
    ----------
    n : int
        Number of locations
    grid_size : float
        Size of the grid
    Returns
    -------
    tuple
        (D, coordx, coordy) where:
        - D is the distance matrix
        - coordx is the x-coordinates of all locations  
        - coordy is the y-coordinates of all locations
    """
    coordy = grid_size *  np.random.random_sample((n,)) # generate random y
    coordx = grid_size *  np.random.random_sample((n,)) # generate random x

    D = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i+1, n):
            d1 = np.array([coordx[i],coordy[i]])
            d2 = np.array([coordx[j],coordy[j]])
            D[i, j] = np.linalg.norm(d1 - d2) + 1
            D[j, i] = D[i, j]
    return D, coordx, coordy

def load_data(cluster_scenario = False, uniform_scenario=False):
    """Load distance matrix and coordinate data from files.
    
    Loads the distance matrix, x and y coordinates, and probability distribution
    for destinations from either the standard dataset or a cluster-based dataset.
    
    Parameters
    ----------
    cluster_scenario : bool, default=False
        Whether to load data from the cluster scenario dataset
        
    Returns
    -------
    tuple
        (D, coordx, coordy, probs) where:
        - D is the distance matrix
        - coordx is the x-coordinates of all locations
        - coordy is the y-coordinates of all locations
        - probs is the probability distribution of destinations
    """
    
    if cluster_scenario:
        D = np.load('data/clusters/D_cluster.npy')
        coordx = np.load('data/clusters/coordx_cluster.npy')
        coordy = np.load('data/clusters/coordy_cluster.npy')
        return D, coordx, coordy, np.ones(D.shape[0])/D.shape[0]
    elif uniform_scenario:
        D = np.load('data/uniform/distance_matrix.npy')
        coordx = np.load('data/uniform/coordx.npy')
        coordy = np.load('data/uniform/coordy.npy')
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
    DynamicQVRPEnv is a custom Gymnasium environment for the Dynamic Vehicle Routing Problem 
    with emissions Quota (QVRP).
    
    This environment simulates a dynamic routing problem where vehicles must be routed to meet 
    demand demands that arrive sequentially over time, while minimizing emissions. 
    The environment supports various configurations including heterogeneous vehicle fleets with 
    different emission profiles, capacity constraints, and emission quotas.
    
    At each time step, a new request arrives, and the agent must decide whether to accept or reject
    the request. If accepted, the request must be assigned to a specific vehicle (when vehicle_assignment=True)
    or automatically inserted into the most suitable route. The environment supports different 
    routing strategies including nearest neighbor insertion and simulated annealing optimization.
    
    Attributes
    ----------
    instance : int
        The current scenario instance being simulated.
    D : np.ndarray
        Distance matrix between all locations.
    coordx : np.ndarray
        X coordinates of all locations for visualization.
    coordy : np.ndarray
        Y coordinates of all locations for visualization.
    p : np.ndarray
        Probability distribution of demand locations.
    all_dests : np.ndarray
        All possible destinations across all scenario instances.
    qs : np.ndarray
        Quantities demanded by each demand in each scenario.
    max_capacity : int
        Maximum capacity of each vehicle.
    total_capacity : int
        Total capacity of the entire fleet.
    T : int
        Number of dynamic requests to be processed (degree of dynamism).
    H : int
        Total horizon length (total number of time steps).
    Q : float
        Emission quota limiting the total emissions allowed.
    hub : int
        Index of the depot/hub location.
    omission_cost : float
        Penalty for rejecting a demand request.
    CO2_penalty : float
        Penalty coefficient for CO2 emissions.
    costs_KM : list
        Operational costs per kilometer for each vehicle type.
    emissions_KM : list
        Emissions per kilometer for each vehicle type.
    re_optimization : bool
        Whether to re-optimize all routes after each new demand acceptance.
    vehicle_assignment : bool
        Whether the agent must explicitly choose which vehicle serves each demand.
    routes : np.ndarray
        Current routes for each vehicle, shape (num_vehicles, max_capacity+2).
    assignment : np.ndarray
        Current assignment of customers to vehicles (0 = rejected).
    t : int
        Current time step (demand index).
    h : int
        Current number of seen dynamic customers.
    remained_capacity : int
        Remaining capacity across all vehicles.
    action_mask : np.ndarray
        Mask indicating which actions are valid at the current step.
    is_O_allowed : np.ndarray
        Mask indicating which customers can be omitted/rejected.
    A : np.ndarray
        Boolean array indicating activated (accepted) destinations.
    NA : np.ndarray
        Boolean array indicating non-activated destinations.
    info : dict
        Dictionary containing information about the current state.
    omitted : list
        List of indices of rejected demand requests.
    episode_reward : float
        Cumulative reward for the current episode.
    
    Methods
    -------
    reset(instance_id=-1, *args, **kwargs)
        Reset the environment to a new state with a specific scenario instance.
    step(action)
        Process the agent's decision for the current demand request.
    _init_instance(instance_id)
        Initialize a new scenario instance.
    _compute_min_med_cap()
        Compute minimum, median distances and remaining capacity 
        for the current state (for observations).
    _get_obs()
        Get the current observation vector.
    sample(H, SA_configs)
        Sample a future state by simulating H steps ahead.
    render(size=100, show_node_num=False, ...)
        Visualize the current state of the environment.
    """
    
    def __init__(self, 
                #  env : AssignmentEnv = None,
                 horizon = 50,
                 Q = 50,
                 DoD = .5,
                 vehicle_capacity = 15, # We assume it homogeneous for all vehicles
                 use_dataset = True,
                 re_optimization  = False,
                 emissions_KM = [.3], 
                 k_min : int = 3,
                 k_med : int = 7,
                 vehicle_assignment = False,
                 costs_KM = None, # for the moment, it has no impact since the focus is on emissions
                 CO2_penalty = 10_000,
                 n_scenarios = None,
                 hub = 0,
                 test = False,
                 allow_initial_omission = True,
                 unknown_p = False,
                 uniforme_p_test = False,
                 noised_p = False,
                 different_quantities = False,
                 cluster_scenario = False,
                 uniform_scenario = False,
                 static_as_dynamic = False,
                 noise_horizon = 0., # Represents the percentage of the noise in horizon. in [0, 1]
                 retain_rate = 0.,
                 seed = 1917,
        ):
        """Initialize the Dynamic QVRP environment.

        Parameters
        ----------
        horizon : int, default=50
            Number of total demands
        Q : float, default=50
            Total emission quota allowed
        DoD : float, default=0.5
            Degree of dynamism - proportion of dynamic requests
        vehicle_capacity : int, default=15
            Capacity of each vehicle (assumed homogeneous)
        retain_rate : float, default=0.0
            Ignore this argument
        use_dataset : bool, default=True
            Whether to use pre-generated dataset
        re_optimization : bool, default=False
            Whether to re-optimize routes after each acceptance
        costs_KM : list, optional
            Cost per km for each vehicle type
        emissions_KM : list, default=[0.3]
            Emissions per km for each vehicle type
        CO2_penalty : float, default=10000
            Penalty coefficient for CO2 emissions (ignore this argument)
        k_min : int, default=3
            Number of nearest neighbors for average distance calculation
        k_med : int, default=7
            Number of nearest neighbors for median distance calculation
        n_scenarios : int, optional
            Number of scenarios to generate
        hub : int, default=0
            Index of depot/hub location
        test : bool, default=False
            Whether in test mode
        allow_initial_omission : bool, default=True
            Whether initial requests can be rejected
        unknown_p : bool, default=False
            Whether destination probabilities are unknown
        uniforme_p_test : bool, default=False
            Whether to use uniform probabilities in test
        noised_p : bool, default=False
            Whether to add noise to probabilities
        different_quantities : bool, default=False
            Whether demands have different quantities
        vehicle_assignment : bool, default=False
            Whether agent must assign vehicles explicitly
        cluster_scenario : bool, default=False
            Whether to use clustered scenarios
        static_as_dynamic : bool, default=False
            Whether to treat static requests as dynamic
        noise_horizon : float, default=0.0
            Noise level for horizon length (0-1)
        seed : int, default=1917
            Random seed
        """
        
        K = horizon
        self.instance = -1
        self.D, self.coordx, self.coordy, self.p = load_data(cluster_scenario, uniform_scenario)
        
        self.emissions_KM = emissions_KM
        if costs_KM is None:
            costs_KM =  np.ones(len(emissions_KM), int).tolist()
            
        self.E = np.array([
            self.emissions_KM[v]*self.D
            for v in range(len(self.emissions_KM))
        ])
        
        np.random.seed(seed)
        
        use_dataset = (test or use_dataset) and not uniform_scenario
        
        if use_dataset and not cluster_scenario:
            retain_comment = f"_retain{retain_rate}" if retain_rate else ""
            scenario_comment = f"_{n_scenarios}" if n_scenarios is not None else ""
            noise_comment = f"noised_" if noised_p else ""
            uniforme = f"_uniforme" if uniforme_p_test else ""
            # with open(f'data/game_K{K}{retain_comment}.pkl', 'rb') as f:
            #     g = pickle.load(f)
            # routes = np.load(f'data/routes_K{K}{retain_comment}.npy')
            if test:
                self.all_dests = np.load(
                    f'data/{noise_comment}destinations_K{K}_100{uniforme}_test.npy'
                ).astype(int)
            else:
                self.all_dests = np.load(
                    f'data/destinations_K{K}{retain_comment}{scenario_comment}{uniforme}.npy'
                ).astype(int)
                
        else:
            if uniform_scenario:
                if test:
                    self.all_dests = np.load(
                        f'data/uniform/destinations_K{K}_100_uniform_test.npy'
                    ).astype(int)
                else:
                    self.all_dests = np.load(
                        f'data/uniform/destinations_K{K}_500_uniform.npy'
                    ).astype(int)
                    
            elif test and cluster_scenario:
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
            if test:
                qs = np.load(f'data/quantities_K{K}.npy')
            else:
                qs = np.random.randint(1, vehicle_capacity//4, (len(self.all_dests), K))
            # np.save(f'data/quantities_K{K}.npy', qs)
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
        self.omission_cost = (2*np.max(self.D) +1)*np.max(self.emissions_KM)
        
        # * Change if obs change
        # self.observation_space = gym.spaces.Box(0, 1, (5+len(emissions_KM),), np.float64) 
        if vehicle_assignment:
            dim_obs = 4 + 2*len(emissions_KM)# if not vehicle_assignment else 5 + len(emissions_KM) + len(self.emissions_KM)
        else:
            dim_obs = 6
        self.observation_space = gym.spaces.Box(0, 1, (dim_obs,), np.float64)
        # self.observation_space = gym.spaces.Box(0, 1, (6,), np.float_)
        
        # * change if actions change
        self.vehicle_assignment = vehicle_assignment
        dim_actions = 2 if not vehicle_assignment else len(self.emissions_KM) + 1 
        self.action_space = gym.spaces.Discrete(dim_actions)
        # self.action_space = gym.spaces.Discrete(2)
        
        self.allow_initial_omission = allow_initial_omission
        
        
    def _init_instance(self, instance_id):
        """Initialize a new scenario instance for the environment.
        
        Sets up the environment with a specific scenario instance, initializing
        routes, assignments, and other state variables.
        
        Parameters
        ----------
        instance_id : int
            The ID of the scenario instance to initialize. If negative, 
            increments the current instance ID.
        """
    
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
            # (self.costs_KM[v] + self.CO2_penalty*self.emissions_KM[v])*self.distance_matrix
            self.emissions_KM[v]*self.distance_matrix
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
        
        
    def _compute_min_med_cap(self):
        """Compute minimum, median distances and the remaining capacity(ies) 
        for the current state.
        
        Calculates two key metrics for the observation space:
        1. The mean emission cost of the k_min nearest neighbors in the current routes
           for each vehicle type
        2. The median distance to the k_med nearest non-activated destinations,
           weighted by their probabilities
        3. The remaining capacity for each vehicle if the vehicle_assignment is True
        otherwise, it returns the remaining capacity for the whole fleet
        
        Returns
        -------
        tuple
            (min_knn, med_knn, cap) where:
            - min_knn is an array of mean emission costs to nearest neighbors for each vehicle
            - med_knn is the median distance to nearest non-activated destinations
            - cap is the remaining capacity for each vehicle if vehicle_assignment=True
            otherwise, it returns the remaining capacity for the whole fleet
        """
        
        if self.static_as_dynamic:
            p = np.ones_like(self.p[self.NA])
        else:
            p = self.p[self.NA].copy()
            # p[~self.NA] = 0
            p /= p.sum()
        
        if self.vehicle_assignment:
            # * The mean of the k nearest neighbors in admitted dests for every vehicle
            alpha = [
                np.where(self.assignment == v)[0]
                for v in range(1, len(self.emissions_KM)+1)
            ]
            masks = [
                np.concatenate([[self.hub], self.dests[alpha[v]]])
                for v in range(len(alpha))
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
            
            # cap = np.full(len(self.emissions_KM), 1.)
            # cap[:self.assignment.max()] -= (
            #     np.bincount(self.assignment)[1:self.assignment.max()+1]
            # )/self.max_capacity
            cap = np.array([
                1. - self.quantities[alpha[v]].sum()/self.max_capacity
                for v in range(len(alpha))
            ])
            
            # print(self.D[masks[0], self.dests[self.t]])
            # print(self.D[masks[1], self.dests[self.t]])
            # print(min_knn)
        else:
            D_A = self.D[self.A, self.dests[self.t]]
            min_knn = np.array([
                max(self.emissions_KM)*np.mean(D_A[knn(D_A, self.k_min)])
                ], np.float64)
            
            cap = np.array([1. - self.remained_capacity/self.total_capacity], np.float64)
        
        # * The mean of the k nearest neighbors in non admitted dests
        D_NA = self.D[self.NA, self.dests[self.t]]
        
        idx_NA = knn(D_NA, self.k_med)
        med_knn = np.median(p[idx_NA]*D_NA[idx_NA])
        # med_knn = np.median(knn(self.D[self.NA, self.dests[self.t]]/(p[self.NA] + 1e-8), self.k_med))
        
        
        return min_knn, med_knn, cap
    
    def _get_obs(self):
        """Get the current observation vector.

        Constructs the observation vector for the current state, including:
        - Quantity demanded by current customer (normalized)
        - Remaining capacity for each vehicle (normalized)
        - Proportion of remaining demands to come
        - Mean emission costs to nearest neighbors for each vehicle (normalized)
        - Median distance to nearest non-activated destinations (normalized)
        - Remaining emission quota (normalized)

        Returns
        -------
        np.ndarray
            The observation vector representing the current state
        """
    
        min_knn, med_knn, cap = self._compute_min_med_cap()
        
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
            # See the _compute_min_med_cap method
            # *self.emissions_KM, # emission of each vehicle, dim = len(self.emissions_KM)
            # * TODO : Maybe find better observations
        ])
        
        return obs
    
    def reset(self, instance_id = -1, *args, **kwargs):
        """Reset the environment to a new state.
    
        Initializes a new scenario instance and returns the initial observation
        and information dictionary.
        
        Parameters
        ----------
        instance_id : int, default=-1
            The ID of the scenario instance to initialize. If negative,
            increments the current instance ID.
        *args, **kwargs
            Additional arguments passed to gym.Env.reset()
        
        Returns
        -------
        tuple
            (observation, info) where:
            - observation is the initial observation vector
            - info is a dictionary containing information about the initial state
        """
        
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
        """Take a step in the environment based on the agent's action.
    
        Processes the agent's decision for the current customer request,
        updates the environment state, and returns the next observation,
        reward, and other information.
        
        Parameters
        ----------
        action : int
            The action to take:
            - 0: Reject the current customer request
            - 1+: Accept and assign to a specific vehicle (if vehicle_assignment=True)
                or just accept (if vehicle_assignment=False)
        
        Returns
        -------
        tuple
            (observation, reward, terminated, truncated, info) where:
            - observation is the next observation vector
            - reward is the reward for the action
            - terminated is True if the episode is done
            - truncated is True if the episode is truncated
            - info is a dictionary containing information about the current state
        """
        
        if (self.h >= self.T-1 or 
            self.remained_capacity <= 0 or
            self.info["remained_quota"] <= 1e-4 
        ):
            print("The episode is done :")
            print(self.info)
            return -1, 0, True, True, self.info
            
        self.h += 1
        assert isinstance(action, (int, np.int_)), f"type : {type(action)}, {action}" # type: ignore
        
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
        """Sample a future state by simulating H steps ahead.
    
        Creates a copy of the current environment and simulates H future
        customer requests, using the provided SA configurations for routing.
        
        Parameters
        ----------
        H : int
            Number of steps to simulate ahead
        SA_configs : dict
            Configuration parameters for the Simulated Annealing algorithm
        
        Returns
        -------
        tuple
            The result of SA_routing2 on the simulated future state
        """
        env = deepcopy(self)
        p = env.p.copy()
        p[env.dests[:self.t+1]] = 0
        p[env.hub] = 0
        
        H = min(H, env.H - env.t - 1)
        
        if len(self.emissions_KM) > 1:
            env.action_mask[env.t : env.t + H+1] = True
            # print(env.t, env.action_mask)
        else:
            env.action_mask = env.is_O_allowed.copy()
            env.action_mask[H+1:] = False
            
        p /= p.sum()
        
        # print(env.is_O_allowed)
        # assert False
        np.random.seed(None)
        future_dests = np.random.choice(len(p), H, False, p)
        # print(env.t, future_dests)
        env.dests[env.t+1 : env.t+H+1] = future_dests
        
        l = [env.hub] + list(env.dests)
        env.mask = np.ix_(l, l)
        env.distance_matrix = env.D[env.mask]
        env.cost_matrix = np.array([
            # (env.costs_KM[v] + env.CO2_penalty*env.emissions_KM[v])*env.distance_matrix
            env.emissions_KM[v]*env.distance_matrix
            for v in range(len(env.emissions_KM))
        ])
        # TODO : implement quantity sampling
        
        # env.action_mask[:self.t+H] = True
        return SA_routing2(env, offline_mode=True, **SA_configs)
    
    def offline_solution(self, *args, **kwargs):
        """Compute an offline solution for the environment.
    
        Creates a copy of the current environment with all customer requests
        known in advance, and computes an optimal routing solution using
        Simulated Annealing.
        
        Parameters
        ----------
        *args, **kwargs
            Additional arguments passed to SA_routing2
        
        Returns
        -------
        tuple
            The result of SA_routing2 on the offline problem
        """
    
        env = deepcopy(self)
        if len(self.cost_matrix) > 1:
            env.action_mask[:] = True
        else:
            env.action_mask = env.is_O_allowed.copy()
        env.T = 0
        return SA_routing2(env, offline_mode=True, *args, **kwargs)
        
    
    def render(self,
               size = 100, show_node_num =False, display_current_node = True,
               display_unactivated = True, display_dests = False,
               color_bar_label = None,
               ):
        """Visualize the current state of the environment.
    
        Creates a graphical representation of the current routes, destinations,
        and other elements of the environment state.
        
        Parameters
        ----------
        size : int, default=100
            Base size for nodes in the visualization
        show_node_num : bool, default=False
            Whether to display node numbers
        display_current_node : bool, default=True
            Whether to highlight the current customer request
        display_unactivated : bool, default=True
            Whether to display non-activated destinations
        display_dests : bool, default=False
            Whether to display all destinations
        color_bar_label : str, default=None
            Label for the color bar (defaults to 'emissions (in kg CO2)')
        
        Returns
        -------
        str
            LaTeX representation of the graph
        """
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
                         edge_cmap=plt.cm.jet, #type:ignore
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
        point = -5
        ax.scatter([point],[point],color=colors[0],label=f'Omitted', s = size, marker='s')
        for i in range(1, len(self.routes)+1):
            ax.scatter([point],[point],color=colors[i],label=f'Vehicle {i}', s = size, marker='s')
        ax.scatter([point],[point],color='black', s = size, marker='s', label='Hub')
        ax.scatter([point],[point],color='white', s = size, marker='s')

        # reverse the order
        plt.draw()
        lgnd = plt.legend(bbox_to_anchor=(1.4, 1.0), loc='upper right')
        # lgnd = plt.legend(loc="lower left", scatterpoints=1, fontsize=10)
        for handle in lgnd.legend_handles:
            handle.set_sizes([50]) #type:ignore
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
    