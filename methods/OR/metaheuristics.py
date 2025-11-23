import numpy as np
from numba import njit
from numba.typed import List
import random
from copy import deepcopy
import multiprocess as mp

# Genetic Algorithm Parameters
POPULATION_SIZE = 500
GENERATIONS = 2_500
MUTATION_RATE = 0.1
ELITISM_COUNT = 7
TOURNAMENT_SIZE = 5
VEHICLE_PENALTY_FACTOR = 0  # Penalty multiplier for exceeding vehicle limit
# EXCESS_PENALTY = 10_000

# njit = lambda x, *args : x

def distribute_random(lst, n):
    """
    Distribute elements of `lst` randomly into `n` sublists.
    Each element is assigned to a bucket uniformly at random, independently.
    Returns a list of `n` lists (some may be empty).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    
    arr = np.array(lst)
    buckets = np.random.randint(0, n, size=len(arr))
    
    # Build result in Python (safe and clear)
    result = [[] for _ in range(n)]
    # result = List([[] for _ in range(n)])
    for item, bucket in zip(arr, buckets):
        result[bucket].append(item)
    
    try:
        out_numba = List()
        for l in result:
            l.append(-1) # To fingerprint the type
            out_numba.append(l)
            out_numba[-1].remove(-1)
    except Exception as e :
        print(result)
        raise(e)
        
    return out_numba

@njit(nogil=True)
def ordered_crossover(parent1, parent2):
    """Numba-optimized ordered crossover"""
    size = len(parent1)
    child = np.empty_like(parent1)
    start, end = sorted(np.random.randint(0, size, 2))
    
    # Copy segment from parent1
    segment = parent1[start:end+1]
    child[start:end+1] = segment
    
    # Fill remaining with parent2's genes
    ptr = 0
    for i in range(size):
        if i < start or i > end:
            while parent2[ptr] in segment:
                ptr += 1
            child[i] = parent2[ptr]
            ptr += 1
    return child

@njit(nogil=True)
def mutate(individual):
    """Numba-optimized swap mutation"""
    if np.random.rand() < MUTATION_RATE:
        i, j = np.random.choice(len(individual), 2, replace=False)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

@njit(nogil=True)
def route_distance(route, distance_matrix):
    """Calculate distance for a route using a specific vehicle's matrix"""
    total = 0.0
    for i in range(len(route)-1):
        total += distance_matrix[route[i], route[i+1]]
    return total

@njit(nogil=True)
def calculate_fitness(individual, demands, distance_matrices, vehicle_capacity, 
                     depot, max_vehicles, Q, omit_penalty):
    """Numba-accelerated fitness calculation with multiple distance matrices"""
    # num_vehicles = distance_matrices.shape[0]
    total_distance = 0.0
    current_load = 0
    # vehicle_used = np.zeros(num_vehicles, dtype=np.bool)
    
    current_route = np.empty(len(individual)+2, dtype=np.int64)
    # current_route = [depot]
    route_length = 0
    routes = []

    v = 0
    oq = 0.
    # served_idx = 0
    # vehicle_assignments = np.empty(len(individual), dtype=np.int64)
    # Split into routes based on capacity
    for i, customer in enumerate(individual):
        demand = demands[customer]
        
        if (total_distance + 
            distance_matrices[v, current_route[route_length], customer] +
            distance_matrices[v, customer, depot] )> Q:
            oq += demands[customer]
            # vehicle_assignments[customer] = -1
            # continue
        
        elif current_load + demand > vehicle_capacity:
            # Finalize current route
            total_distance += distance_matrices[v, current_route[route_length], depot]
            current_route[route_length] = depot
            routes.append(current_route[:route_length+1])
            # Start new route
            
            
            current_load = demand
            if v+1 >= len(distance_matrices):
                oq += np.sum(demands[individual[i+1:]])
                # print(individual[i+1:])
                break
            
            v+=1
            current_route[0] = depot
            current_route[1] = customer
            total_distance += distance_matrices[v, depot, customer]
            route_length = 1
                
            # vehicle_assignments[customer] = v
        else:
            total_distance += distance_matrices[v, current_route[route_length], customer]
            route_length += 1
            current_route[route_length] = customer
            current_load += demand
            
            # vehicle_assignments[customer] = v
    
    # oq += np.sum(demands[i:])
    # Add final route
    current_route[route_length] = depot
    routes.append(current_route[:route_length+1])

    # # Calculate distances and vehicle assignments
    # vehicle_assignments = np.empty(len(routes), dtype=np.int64)
    # remaining_vehicles = List(range(num_vehicles))
    # for i, route in enumerate(routes):
    #     min_dist = np.inf
    #     best_vehicle = 0
    #     # Find best vehicle for this route
    #     for v in remaining_vehicles:
    #         dist = route_distance(route, distance_matrices[v])
    #         if dist < min_dist:
    #             min_dist = dist
    #             best_vehicle = v
    #     total_distance += min_dist
    #     vehicle_assignments[i] = best_vehicle
    #     vehicle_used[best_vehicle] = True
    #     remaining_vehicles.remove(best_vehicle)

    # Calculate vehicle count penalty
    # used_count = np.sum(vehicle_used)
    # if used_count > max_vehicles:
    #     penalty = (used_count - max_vehicles) * VEHICLE_PENALTY_FACTOR * distance_matrices.mean()
    # else:
    #     penalty = 0.0
        
    # TODO update the penalty to adapt to the hard constraint
    # penalty += CO2_penalty*max(0, total_distance - Q)
    penalty = omit_penalty*oq

    return total_distance + penalty#, oq

class MultiVehicleVRPSolver:
    def __init__(self, distance_matrices, demands, vehicle_capacity, max_vehicles, 
                 Q = 100, depot=0, #excess_penalty = 10_000
        ):
        """
        Initialize with:
        - distance_matrices: 3D numpy array (vehicles x nodes x nodes)
        - demands: 1D array of node demands
        - vehicle_capacity: maximum load per vehicle
        - max_vehicles: maximum number of distinct vehicles allowed
        - depot: depot node index
        """
        self.distance_matrices = distance_matrices.astype(np.float64)
        self.demands = demands.astype(np.int64)
        self.vehicle_capacity = vehicle_capacity
        self.max_vehicles = max_vehicles
        self.omit_penalty = np.amax(distance_matrices)*3
        print(self.omit_penalty)
        # self.excess_penalty = excess_penalty
        self.Q = Q
        self.depot = depot
        self.num_nodes = len(demands)
        self.customer_indices = np.array([i for i in range(self.num_nodes) if i != depot], dtype=np.int64)
        self.num_vehicles = distance_matrices.shape[0]

    def create_individual(self):
        """Create random customer permutation"""
        ind = self.customer_indices.copy()
        np.random.shuffle(ind)
        return ind

    def initial_population(self):
        return [self.create_individual() for _ in range(POPULATION_SIZE)]

    @staticmethod
    @njit
    def evaluate_population(population, demands, distance_matrices, vehicle_capacity, 
                            depot, max_vehicles, Q, omit_penalty):
        fitness = np.empty(len(population), dtype=np.float64)
        for i in range(len(population)):
            fitness[i] = calculate_fitness(population[i], demands, distance_matrices,
                                          vehicle_capacity, depot, max_vehicles, Q, omit_penalty)
        return fitness

    @staticmethod
    @njit
    def evolve_population(population, fitness, demands, distance_matrices, 
                         vehicle_capacity, depot, max_vehicles):
        new_pop = np.empty_like(population)
        elite_indices = np.argsort(fitness)[:ELITISM_COUNT]
        new_pop[:ELITISM_COUNT] = population[elite_indices]
        
        for i in range(ELITISM_COUNT, POPULATION_SIZE):
            # Tournament selection
            tournament = np.random.choice(len(population), TOURNAMENT_SIZE)
            best_idx = tournament[np.argmin(fitness[tournament])]
            parent1 = population[best_idx]
            
            tournament = np.random.choice(len(population), TOURNAMENT_SIZE)
            best_idx = tournament[np.argmin(fitness[tournament])]
            parent2 = population[best_idx]
            
            # Genetic operations
            child = ordered_crossover(parent1, parent2)
            child = mutate(child)
            new_pop[i] = child
        
        return new_pop

    def solve(self):
        population = np.array(self.initial_population())
        best = population[0]
        best_score = np.inf
        fitness = self.evaluate_population(population, self.demands, self.distance_matrices,
                                         self.vehicle_capacity, self.depot, self.max_vehicles,
                                         self.Q, self.omit_penalty)
        
        for gen in range(GENERATIONS):
            population = self.evolve_population(population, fitness, self.demands,
                                              self.distance_matrices, self.vehicle_capacity,
                                              self.depot, self.max_vehicles)
            fitness = self.evaluate_population(population, self.demands,
                                             self.distance_matrices, self.vehicle_capacity,
                                             self.depot, self.max_vehicles, self.Q, self.omit_penalty)
            
            best_idx = np.argmin(fitness)
            print(f"Generation {gen+1}: Best Score = {fitness[best_idx]:.2f}")
            if fitness[best_idx] < best_score:
                best_score = fitness[best_idx]
                best = population[best_idx]
        
        # best_idx = np.argmin(fitness)
        # return population[best_idx], fitness[best_idx]
        return best, best_score

    def decode_solution(self, individual):
        """Decode solution with vehicle assignments"""
        current_route = [self.depot]
        current_load = 0
        routes = []
        total_distance = 0.
        # route_length = 0
        v = 0
        oq = 0.
        
        vehicle_assignments = np.empty(len(individual)+1, dtype=np.int64)
        # Split into routes based on capacity
        for i, customer in enumerate(individual):
            demand = self.demands[customer]

            if (total_distance + 
                self.distance_matrices[v, current_route[-1], customer] +
                self.distance_matrices[v, customer, self.depot] )> self.Q:
                oq += demand
                vehicle_assignments[customer] = -1
                # continue

            elif current_load + demand > self.vehicle_capacity:
                # Finalize current route
                total_distance += self.distance_matrices[v, current_route[-1], self.depot]
                current_route.append(self.depot)
                routes.append(np.array(current_route, dtype=np.int64))
                
                # route_length = 1
                current_load = demand
                if v+1 >= self.max_vehicles:
                    oq += np.sum(self.demands[individual[i+1:]])
                    print(individual[i+1:])
                    break
                
                v+=1
                # Start new route
                current_route = [self.depot, customer]
                total_distance += self.distance_matrices[v, self.depot, customer]
                vehicle_assignments[customer] = v
            else:
                total_distance += self.distance_matrices[v, current_route[-1], customer]
                # route_length += 1
                current_route.append(customer)
                current_load += demand

                vehicle_assignments[customer] = v
          
        
        total_distance += self.distance_matrices[v, current_route[-1], self.depot]
        current_route.append(self.depot)      
        
        routes.append(np.array(current_route, dtype=np.int64))
        print('excess : ', total_distance - self.Q)
        
        # # Split into routes
        # full_routes = []
        # for customer in individual:
        #     demand = self.demands[customer]
        #     if current_load + demand > self.vehicle_capacity:
        #         current_route.append(self.depot)
        #         full_routes.append(np.array(current_route, dtype=np.int64))
        #         current_route = [self.depot, customer]
        #         current_load = demand
        #     else:
        #         current_route.append(customer)
        #         current_load += demand
        # current_route.append(self.depot)
        # full_routes.append(np.array(current_route, dtype=np.int64))
        
        # # Assign best vehicles
        # decoded_routes = []
        # vehicle_assignments = []
        # remaining_vehicles = list(range(self.num_vehicles))
        # for route in full_routes:
        #     min_dist = np.inf
        #     best_vehicle = 0
        #     for v in remaining_vehicles:
        #         dist = route_distance(route, self.distance_matrices[v])
        #         if dist < min_dist:
        #             min_dist = dist
        #             best_vehicle = v
        #     decoded_routes.append(route)
        #     vehicle_assignments.append(best_vehicle)
        #     remaining_vehicles.remove(best_vehicle)
        
        return routes, vehicle_assignments, oq

# Example usage
def genetic(D, qs, capacity, emissions_KM):
    # Problem parameters
    NUM_NODES = len(D)
    VEHICLE_CAPACITY = capacity
    MAX_VEHICLES = len(emissions_KM)
    NUM_VEHICLES = len(emissions_KM)  # Total available vehicle types
    # Q = 100
    # excess_penalty = 10_000
    
    # # Generate multiple distance matrices
    # distance_matrices = np.random.randint(1, 100, (NUM_VEHICLES, NUM_NODES, NUM_NODES)).astype(np.float64)
    # for v in range(NUM_VEHICLES):
    #     np.fill_diagonal(distance_matrices[v], 0)
    #     distance_matrices[v] = (distance_matrices[v] + distance_matrices[v].T) / 2
    
    # distance_matrix = np.load('data/distance_matrix.npy').astype(np.float64)#[:100, :100]
    # scenarios = np.load('data/destinations_K50_100_test.npy').astype(np.int64)
    # s = [0] + list(scenarios[s_idx])
    # mask = np.ix_(s, s)
    # distance_matrix = distance_matrix[mask]
    
    # em_factors = [.1, .3]
    Ds = np.array([
        emissions_KM[v]*D
        for v in range(NUM_VEHICLES)
    ]).astype(np.float64)
    
    # print(distance_matrices.shape)
    
    # Generate demands
    demands = np.zeros(NUM_NODES, dtype=np.int64)
    # demands[1:] = np.random.randint(0, 2, NUM_NODES-1)
    demands[1:] = qs #np.ones(NUM_NODES-1)
    
    # Initialize and run solver
    solver = MultiVehicleVRPSolver(
        distance_matrices=Ds,
        demands=demands,
        vehicle_capacity=VEHICLE_CAPACITY,
        max_vehicles=MAX_VEHICLES
    )
    
    best_solution, best_score = solver.solve()
    routes, assignment, oq = solver.decode_solution(best_solution)
    
    # print("\nBest solution:")
    # print(f"Total score: {best_score:.2f}")
    # print(f"Vehicles assignments: {assignment+1}")
    lengths = 0
    for i, route in enumerate(routes):
        lengths += len(route) -2 
        # print(f"Route {i+1} (Vehicle {i+1}): {route.tolist()}")
    # print(f"Total omitted: {oq:.2f}")
    # print(f"Total rewards: {NUM_NODES-1 - oq:.2f}")
    assert lengths == NUM_NODES-1 - oq
    # print(f"lenghts: {lenghts:.2f}")
    
    return routes, assignment+1, lengths
    

def multiple_genetic(D, qs, capacity, emissions_KM, n_sample = 5):
    
    
    def process(i, q):
        # a = dests[i_id][np.where(a_GTS == 0)].astype(int)
        
        np.random.seed(i)
        res = dict()
        routes, assignment, r = genetic(D, qs, capacity, emissions_KM)
        res['a'] = assignment
        res['routes'] = routes
        res['r'] = r
        # res['info'] = [info]
        q.put((i, res))
        # print(f'DP {i} done')
        return
    
    episode_rewards = np.zeros(n_sample)
    actions = [[] for _ in range(n_sample)]
    routes = [[] for _ in range(n_sample)]
    
    q = mp.Manager().Queue()
    
    ps = []
    for i in range(n_sample):
        ps.append(
            mp.Process(target = process, args = (i, q, ))
        )
        ps[-1].start()
    # p[4*i+3].start()
    
    for p in ps:
        p.join()
        
    while not q.empty():
        i, d = q.get()
        episode_rewards[i] = d["r"]
        actions[i] = d["a"]
        routes[i] = d["routes"]
        
    best_idx = np.argmax(episode_rewards)
        
    return routes[best_idx], actions[best_idx], episode_rewards[best_idx]


@njit
def calculate_cost(permutation, demands, distance_matrices, vehicle_capacity, 
                     depot, max_vehicles, Q, omit_penalty):
    """Numba-accelerated fitness calculation with multiple distance matrices"""
    # num_vehicles = distance_matrices.shape[0]
    total_distance = 0.0
    current_load = 0
    # vehicle_used = np.zeros(num_vehicles, dtype=np.bool)
    
    # current_route = np.zeros(len(permutation)+2, dtype=np.int64)
    current_route = np.zeros(vehicle_capacity+2, dtype=np.int64)
    # current_route = [depot]
    route_length = 0
    # routes = []
    routes = np.zeros((max_vehicles, vehicle_capacity+2), dtype=np.int64)#List(List())#[[]]#
    v = 0
    oq = 0.
    
    # served_idx = 0
    # vehicle_assignments = np.empty(len(individual), dtype=np.int64)
    # Split into routes based on capacity
    for i, customer in enumerate(permutation):
        demand = demands[customer]
        
        if not demand:
            continue
        
        if (total_distance + 
            distance_matrices[v, current_route[route_length], customer] +
            distance_matrices[v, customer, depot] )> Q:
            oq += demand#demands[customer]
            # print()
            # vehicle_assignments[customer] = -1
            # continue
        
        elif current_load + demand > vehicle_capacity:
            # Finalize current route
            total_distance += distance_matrices[v, current_route[route_length], depot]
            # current_route[route_length] = depot
            routes[v, :] = current_route[:]
            # routes[v, :route_length+1] = current_route[:route_length+1]
            # Start new route
            
            
            current_load = demand
            if v+1 >= max_vehicles-1:
                # print(v)
                oq += np.sum(demands[permutation[i:]])
                # print(individual[i+1:])
                break
            
            v+=1
            current_route[:] = 0
            current_route[0] = depot
            current_route[1] = customer
            total_distance += distance_matrices[v, depot, customer]
            route_length = 1
                
            # vehicle_assignments[customer] = v
        else:
            total_distance += distance_matrices[v, current_route[route_length], customer]
            route_length += 1
            current_route[route_length] = customer
            current_load += demand
            
            # vehicle_assignments[customer] = v
    
    total_distance += distance_matrices[v, current_route[route_length], depot]
    # current_route[route_length] = depot
    # routes[v, :route_length+2] = current_route[:route_length+2]
    routes[v, :] = current_route[:]
    # routes.append(current_route[:route_length+1])

    penalty = omit_penalty*oq

    # return total_distance + penalty#, oq
    return total_distance + penalty, oq, routes

@njit
def calculate_cost2(sol, demands, distance_matrices, vehicle_capacity, 
                     depot, max_vehicles, Q, omit_penalty):
    """Numba-accelerated fitness calculation with multiple distance matrices"""
    # num_vehicles = distance_matrices.shape[0]
    total_distance = 0.0
    # vehicle_used = np.zeros(num_vehicles, dtype=np.bool)
    
    # current_route = np.zeros(len(permutation)+2, dtype=np.int64)
    # current_route = [depot]
    # routes = []
    routes = np.zeros((max_vehicles, vehicle_capacity+2), dtype=np.int64)#List(List())#[[]]#
    oq = 0.
    
    # served_idx = 0
    # vehicle_assignments = np.empty(len(individual), dtype=np.int64)
    # Split into routes based on capacity
    for v, route in enumerate(sol):
        current_load = 0
        route_length = 0
        current_route = np.zeros(vehicle_capacity+2, dtype=np.int64)
        current_route[0] = depot
        
        for i, customer in enumerate(route):
            demand = demands[customer]
            
            if not demand:
                continue
            
            if (total_distance + 
                distance_matrices[v, current_route[route_length], customer] +
                distance_matrices[v, customer, depot] )> Q:
                oq += demand#demands[customer]
                # print()
                # vehicle_assignments[customer] = -1
                # continue
            
            elif current_load + demand > vehicle_capacity:
                # Finalize current route
                total_distance += distance_matrices[v, current_route[route_length], depot]
                # current_route[route_length] = depot
                routes[v, :] = current_route[:].copy()
                # routes[v, :route_length+1] = current_route[:route_length+1]
                # Start new route
                
                
                # current_load = demand
                # oq += np.sum(demands[route[i:]])
                oq += demand#demands[customer]
                
                    
                # vehicle_assignments[customer] = v
            else:
                total_distance += distance_matrices[v, current_route[route_length], customer]
                route_length += 1
                current_route[route_length] = customer
                current_load += demand
            
            # vehicle_assignments[customer] = v
    
    total_distance += distance_matrices[v, current_route[route_length], depot]
    # current_route[route_length] = depot
    # routes[v, :route_length+2] = current_route[:route_length+2]
    routes[v, :] = current_route[:]
    # routes.append(current_route[:route_length+1])

    penalty = omit_penalty*oq

    # return total_distance + penalty#, oq
    return total_distance + penalty, oq, routes



@njit
def calculate_routes_and_assignment(permutation, demands, distance_matrices, vehicle_capacity, 
                     depot, max_vehicles, Q, omit_penalty):
    """Numba-accelerated fitness calculation with multiple distance matrices"""
    # num_vehicles = distance_matrices.shape[0]
    total_distance = 0.0
    distances = np.zeros(len(distance_matrices))
    
    current_load = 0
    # vehicle_used = np.zeros(num_vehicles, dtype=np.bool)
    
    # current_route = np.zeros(len(permutation)+2, dtype=np.int64)
    current_route = np.zeros(vehicle_capacity+2, dtype=np.int64)
    assignment = np.zeros(len(demands), np.int64)
    # current_route = [depot]
    route_length = 0
    # routes = []
    routes = np.zeros((max_vehicles, vehicle_capacity+2), dtype=np.int64)#List(List())#[[]]#
    v = 0
    oq = 0.
    
    # served_idx = 0
    # vehicle_assignments = np.empty(len(individual), dtype=np.int64)
    # Split into routes based on capacity
    for i, customer in enumerate(permutation):
        demand = demands[customer]
        
        if not demand:
            continue
        
        if (total_distance + 
            distance_matrices[v, current_route[route_length], customer] +
            distance_matrices[v, customer, depot] )> Q:
            oq += demand#demands[customer]
            assignment[customer] = 0
            # print()
            # vehicle_assignments[customer] = -1
            # continue
        
        elif current_load + demand > vehicle_capacity:
            # Finalize current route
            total_distance += distance_matrices[v, current_route[route_length], depot]
            distances[v] += distance_matrices[v, current_route[route_length], depot]
            # current_route[route_length] = depot
            routes[v, :] = current_route[:]
            # routes[v, :route_length+1] = current_route[:route_length+1]
            # Start new route
            
            
            current_load = demand
            if v+1 >= max_vehicles-1:
                # print(v)
                oq += np.sum(demands[permutation[i:]])
                # print(individual[i+1:])
                break
            
            v+=1
            current_route[:] = 0
            current_route[0] = depot
            current_route[1] = customer
            assignment[customer] = v+1
            total_distance += distance_matrices[v, depot, customer]
            distances[v] += distance_matrices[v, depot, customer]
            route_length = 1
                
            # vehicle_assignments[customer] = v
        else:
            total_distance += distance_matrices[v, current_route[route_length], customer]
            distances[v] += distance_matrices[v, current_route[route_length], customer]
            route_length += 1
            current_route[route_length] = customer
            assignment[customer] = v+1
            current_load += demand
            
            # vehicle_assignments[customer] = v
    
    # * It must calculate the return to the depot even if the capacity is not reached
    # * This was the source of a bug
    total_distance += distance_matrices[v, current_route[route_length], depot]
    distances[v] += distance_matrices[v, current_route[route_length], depot]
    
    # current_route[route_length] = depot
    # routes[v, :route_length+2] = current_route[:route_length+2]
    routes[v, :] = current_route[:]
    # routes.append(current_route[:route_length+1])

    # penalty = omit_penalty*oq

    # return total_distance + penalty#, oq
    return distances, oq, routes, assignment[1:] # exclude the hub

@njit
def calculate_routes_and_assignment2(sol, demands, distance_matrices, vehicle_capacity, 
                     depot, max_vehicles, Q, omit_penalty):
    """Numba-accelerated fitness calculation with multiple distance matrices"""
    # num_vehicles = distance_matrices.shape[0]
    total_distance = 0.0
    distances = np.zeros(len(distance_matrices))
    
    # vehicle_used = np.zeros(num_vehicles, dtype=np.bool)
    
    # current_route = np.zeros(len(permutation)+2, dtype=np.int64)
    assignment = np.zeros(len(demands), np.int64)
    # current_route = [depot]
    # routes = []
    routes = np.zeros((max_vehicles, vehicle_capacity+2), dtype=np.int64)#List(List())#[[]]#
    oq = 0.
    
    # served_idx = 0
    # vehicle_assignments = np.empty(len(individual), dtype=np.int64)
    for v, route in enumerate(sol):
        # Split into routes based on capacity
        current_load = 0
        route_length = 0
        current_route = np.zeros(vehicle_capacity+2, dtype=np.int64)
        current_route[0] = depot
        for i, customer in enumerate(route):
            demand = demands[customer]
            
            if not demand:
                continue
            
            if (total_distance + 
                distance_matrices[v, current_route[route_length], customer] +
                distance_matrices[v, customer, depot] )> Q:
                oq += demand#demands[customer]
                assignment[customer] = 0
                # print()
                # vehicle_assignments[customer] = -1
                # continue
            
            elif current_load + demand > vehicle_capacity:
                # Finalize current route
                total_distance += distance_matrices[v, current_route[route_length], depot]
                distances[v] += distance_matrices[v, current_route[route_length], depot]
                # current_route[route_length] = depot
                routes[v, :] = current_route[:].copy()
                # routes[v, :route_length+1] = current_route[:route_length+1]
                # Start new route
                
                
                # current_load = demand
                # oq += np.sum(demands[route[i:]])
                assignment[customer] = 0
                oq += demand#demands[customer]
            else:
                total_distance += distance_matrices[v, current_route[route_length], customer]
                distances[v] += distance_matrices[v, current_route[route_length], customer]
                route_length += 1
                current_route[route_length] = customer
                assignment[customer] = v+1
                current_load += demand
                
                # vehicle_assignments[customer] = v
        
    # * It must calculate the return to the depot even if the capacity is not reached
    # * This was the source of a bug
    total_distance += distance_matrices[v, current_route[route_length], depot]
    distances[v] += distance_matrices[v, current_route[route_length], depot]
    
    # current_route[route_length] = depot
    # routes[v, :route_length+2] = current_route[:route_length+2]
    routes[v, :] = current_route[:]
    # routes.append(current_route[:route_length+1])

    # penalty = omit_penalty*oq

    # return total_distance + penalty#, oq
    return distances, oq, routes, assignment[1:] # exclude the hub


@njit
def generate_neighbor(current_solution):
    """Generate neighboring solution using various operators"""
    new_solution = current_solution.copy()
    n = len(current_solution)
    
    # Choose between different neighborhood operations
    rand_val = np.random.random()
    
    if rand_val < 0.6:  # Swap
        i1, i2 = np.random.randint(0, n, size=2)
        new_solution[i1], new_solution[i2] = new_solution[i2], new_solution[i1]
    elif rand_val < 0.7:  # Reverse
        start, end = sorted(np.random.randint(0, n, size=2))
        new_solution[start:end+1] = new_solution[start:end+1][::-1]
    else:  # Insert
        pos = np.random.randint(0, n)
        customer = new_solution[pos]
        new_solution = np.delete(new_solution, pos)
        insert_pos = np.random.randint(0, len(new_solution)+1)
        new_solution = np.array(
            list(new_solution[:insert_pos]) + [customer] + list(new_solution[insert_pos:]))
        # new_solution = np.insert(new_solution, insert_pos, customer)
    
    return new_solution

@njit
def generate_neighbor2(current_solution, cap):
    """Generate neighboring solution using various operators"""
    new_solution = current_solution.copy()
    # n = len(current_solution)
    a = np.array([len(l) for l in current_solution])
    
    non_empty = np.where(a > 0)[0]
    non_full = np.where(a < cap)[0]
    # for v in range(n):
    #     if len(current_solution[v]) < cap:
    #         non_full.append(v)
            
    non_empty = np.array(list(non_empty), dtype=np.int64)
    # Choose between different neighborhood operations
    rand_val = np.random.random()
    if len(non_full) == 0:
        rand_val -= .601
    if rand_val < 0.3:  # Swap
        v1, v2 = np.random.choice(non_empty, size=2, replace = True)
        i1 = np.random.randint(0, len(current_solution[v1]))
        i2 = np.random.randint(0, len(current_solution[v2]))
        new_solution[v1][i1], new_solution[v2][i2] = new_solution[v2][i2], new_solution[v1][i1]
    elif rand_val < 0.4:  # Reverse
        v1 = np.random.choice(non_empty, replace = True)
        start, end = sorted(np.random.randint(0, len(current_solution[v1]), size=2))
        new_solution[v1][start:end+1] = new_solution[v1][start:end+1][::-1]
    else:  # Insert
        v1 = np.random.choice(non_empty)
        v2 = np.random.choice(non_full)
        pos = np.random.randint(0, len(current_solution[v1]))
        customer = new_solution[v1].pop(pos)
        insert_pos = np.random.randint(0, len(current_solution[v2])+1)
        new_solution[v2].insert(insert_pos, customer)
        # new_solution[v2] = new_solution[v2][:insert_pos] + List([customer]) + new_solution[v2][insert_pos:]
        # new_solution = np.insert(new_solution, insert_pos, customer)
    
    return new_solution


@njit
def simulated_annealing_vrp(D, demands, capacity, initial_solution, 
                            max_vehicles=5, initial_temp=10_000.0, cooling_rate=0.995,
                           max_iter=10_000, depot = 0,
                           Q = 100):
    """Numba-optimized SA for multi-vehicle VRP"""
    # numba.seed(seed)
    # np.random.seed(seed)
    # Problem setup
    # customers = np.arange(1, n + 1)
    dist_mat = D#compute_distance_matrix(coords)
    
    omit_penalty = 3*np.amax(D)
    
    # Initial solution
    current_solution = initial_solution
    # current_cost, current_vehicles, _ = calculate_total_cost(
    #     current_solution, demands, capacity, dist_mat, max_vehicles
    # )
    current_cost, current_oq, _ = calculate_cost(current_solution, demands, dist_mat, capacity, 
                     depot, max_vehicles, Q, omit_penalty)
    best_solution = current_solution.copy()
    best_cost = current_cost
    # best_vehicles = current_vehicles
    best_oq = current_oq
    
    # We shake up the solution for better exploration
    # current_solution = np.random.permutation(current_solution).astype(np.int64)
    
    # SA parameters
    T = initial_temp
    
    for i in range(max_iter):
        new_solution = generate_neighbor(current_solution)
        new_cost, oq, _ = calculate_cost(new_solution, demands, dist_mat, capacity, 
                     depot, max_vehicles, Q, omit_penalty)
        # new_cost, new_vehicles, _ = calculate_total_cost(
        #     new_solution, demands, capacity, dist_mat, max_vehicles
        # )
        
        # Acceptance criteria with vehicle count consideration
        if (new_cost < current_cost or 
            np.exp((current_cost - new_cost)/T) > np.random.rand()):
            current_solution = new_solution
            current_cost = new_cost
            current_oq = oq
            
            if oq <= best_oq:
                if current_cost < best_cost:
                    best_solution = current_solution.copy()
                    best_cost = new_cost
                    best_oq = oq
        
        # Adaptive cooling
        if T > 1e-10:
            T *= cooling_rate
            if i % 1000 == 0:
                T *= 1.2  # Extra heating boost
    
    # Get final routes
    # print(best_solution)
    # print(best_solution.shape)
    best_costs, best_oq, best_routes, assignment = calculate_routes_and_assignment(
        best_solution, demands, dist_mat, capacity, 
        depot, max_vehicles, Q, omit_penalty
    )
    # _, _, best_routes = calculate_total_cost(
    #     best_solution, demands, capacity, dist_mat, max_vehicles
    # )
    
    # print('Final temperature : ', T)
    
    return best_routes, best_costs, best_oq, assignment

@njit
def simulated_annealing_vrp2(D, demands, capacity, initial_solution, 
                            max_vehicles=5, initial_temp=10_000.0, cooling_rate=0.995,
                           max_iter=10_000, depot = 0,
                           Q = 100):
    """Numba-optimized SA for multi-vehicle VRP"""
    # numba.seed(seed)
    # np.random.seed(seed)
    # Problem setup
    # customers = np.arange(1, n + 1)
    dist_mat = D#compute_distance_matrix(coords)
    
    omit_penalty = 3*np.amax(D)
    
    # Initial solution
    current_solution = initial_solution
    # current_cost, current_vehicles, _ = calculate_total_cost(
    #     current_solution, demands, capacity, dist_mat, max_vehicles
    # )
    current_cost, current_oq, _ = calculate_cost2(current_solution, demands, dist_mat, capacity, 
                     depot, max_vehicles, Q, omit_penalty)
    best_solution = current_solution.copy()
    best_cost = current_cost
    # best_vehicles = current_vehicles
    best_oq = current_oq
    
    # We shake up the solution for better exploration
    # current_solution = np.random.permutation(current_solution).astype(np.int64)
    
    # SA parameters
    T = initial_temp
    
    for i in range(max_iter):
        new_solution = generate_neighbor2(current_solution, capacity)
        new_cost, oq, _ = calculate_cost2(new_solution, demands, dist_mat, capacity, 
                     depot, max_vehicles, Q, omit_penalty)
        # new_cost, new_vehicles, _ = calculate_total_cost(
        #     new_solution, demands, capacity, dist_mat, max_vehicles
        # )
        
        # Acceptance criteria with vehicle count consideration
        if (new_cost < current_cost or 
            np.exp((current_cost - new_cost)/T) > np.random.rand()):
            current_solution = new_solution
            current_cost = new_cost
            current_oq = oq
            
            if oq <= best_oq:
                if current_cost < best_cost:
                    best_solution = current_solution.copy()
                    best_cost = new_cost
                    best_oq = oq
        
        # Adaptive cooling
        if T > 1e-10:
            T *= cooling_rate
            if i % 1000 == 0:
                T *= 1.2  # Extra heating boost
    
    # Get final routes
    # print(best_solution)
    # print(best_solution.shape)
    best_costs, best_oq, best_routes, assignment = calculate_routes_and_assignment2(
        best_solution, demands, dist_mat, capacity, 
        depot, max_vehicles, Q, omit_penalty
    )
    # _, _, best_routes = calculate_total_cost(
    #     best_solution, demands, capacity, dist_mat, max_vehicles
    # )
    
    # print('Final temperature : ', T)
    
    return best_routes, best_costs, best_oq, assignment


@njit
def simulated_annealing_tsp(D, demands, capacity, initial_solution, 
                            initial_temp=100.0, cooling_rate=0.99,
                           max_iter=500, depot = 0,
                           Q = 100):
    """Numba-optimized SA for multi-vehicle VRP"""
    # numba.seed(seed)
    # Problem setup
    # customers = np.arange(1, n + 1)
    dist_mat = D#compute_distance_matrix(coords)
    
    omit_penalty = 3*np.amax(D)
    
    # Initial solution
    current_solution = initial_solution
    # current_cost, current_vehicles, _ = calculate_total_cost(
    #     current_solution, demands, capacity, dist_mat, max_vehicles
    # )
    current_cost, current_oq, _ = calculate_cost(current_solution, demands, dist_mat, capacity, 
                     depot, 1, Q, omit_penalty)
    best_solution = current_solution.copy()
    best_cost = current_cost
    # best_vehicles = current_vehicles
    best_oq = current_oq
    
    # We shake up the solution for better exploration
    # current_solution = np.random.permutation(current_solution).astype(np.int64)
    
    # SA parameters
    T = initial_temp
    
    for i in range(max_iter):
        new_solution = generate_neighbor(current_solution)
        new_cost, oq, _ = calculate_cost(new_solution, demands, dist_mat, capacity, 
                     depot, 1, Q, omit_penalty)
        # new_cost, new_vehicles, _ = calculate_total_cost(
        #     new_solution, demands, capacity, dist_mat, max_vehicles
        # )
        
        # Acceptance criteria with vehicle count consideration
        if (new_cost < current_cost or 
            np.exp((current_cost - new_cost)/T) > np.random.rand()):
            current_solution = new_solution
            current_cost = new_cost
            current_oq = oq
            
            if oq <= best_oq:
                if current_cost < best_cost:
                    best_solution = current_solution.copy()
                    best_cost = new_cost
                    best_oq = oq
        
        # Adaptive cooling
        if T > 1e-10:
            T *= cooling_rate
            if i % 1000 == 0:
                T *= 1.2  # Extra heating boost
    
    # Get final routes
    # print(best_solution)
    # print(best_solution.shape)
    best_costs, best_oq, best_routes, assignment = calculate_routes_and_assignment(
        best_solution, demands, dist_mat, capacity, 
        depot, 1, Q, omit_penalty
    )
    # _, _, best_routes = calculate_total_cost(
    #     best_solution, demands, capacity, dist_mat, max_vehicles
    # )
    
    # print('Final temperature : ', T)
    
    return best_routes, best_costs, best_oq, assignment

# Example usage
def SA_vrp(distance_matrix, Q, qs, capacity, emissions_KM, 
           customers = None, log = False,
           initial_solution = None,
           SA_configs = dict(
              initial_temp=1_000,
              cooling_rate=0.995,
              max_iter=50_000, 
            ),
    ):
    
    # Problem parameters
    # em_factors = [.1, .1, .3, .3]
    # em_factors = [.1, .3]
    max_vehicles = len(emissions_KM)
    # Q = 100
    # excess_penalty = 10_000
    
    # # Generate multiple distance matrices
    # distance_matrices = np.random.randint(1, 100, (NUM_VEHICLES, NUM_NODES, NUM_NODES)).astype(np.float64)
    # for v in range(NUM_VEHICLES):
    #     np.fill_diagonal(distance_matrices[v], 0)
    #     distance_matrices[v] = (distance_matrices[v] + distance_matrices[v].T) / 2
    
    # distance_matrix = np.load('data/distance_matrix.npy').astype(np.float64)#[:100, :100]
    # scenarios = np.load('data/destinations_K100_100_test.npy').astype(np.int64)
    
    # s = [0] + list(scenarios[s_idx])
    NUM_NODES = len(distance_matrix)
    # mask = np.ix_(s, s)
    # distance_matrix = distance_matrix[mask]
    
    D = np.array([
        emissions_KM[v]*distance_matrix
        for v in range(max_vehicles)
    ]).astype(np.float64)
    
    
    # print(distance_matrices.shape)
    
    # # Generate demands
    # demands = np.zeros(NUM_NODES, dtype=np.int64)
    # # demands[1:] = np.random.randint(0, 2, NUM_NODES-1)
    # demands[1:] = np.ones(NUM_NODES-1)
    
    # # Generate sample data
    # np.random.seed(1917)
    # depot = np.array([[0, 0]])
    # customers = np.random.randint(0, 100, (20, 2))
    # coords = np.vstack((depot, customers))
    # emission = [.1, .3, .3]
    # # Problem setup
    # # n = coords.shape[0] - 1  # Exclude depot
    # # customers = np.arange(1, n + 1)
    # mat = compute_distance_matrix(coords)
    # D = np.array([
    #     mat*emission[v]
    #     for v in range(len(emission))
    # ])
    
    
    # demands = np.concatenate(([0], np.random.randint(1, 5, 20)))
    # Generate demands
    demands = np.zeros(NUM_NODES, dtype=np.int64)
    n = D.shape[1] - 1  # Exclude depot
    customers = np.arange(1, n + 1, dtype=np.int64) if customers is None else customers
    
    if initial_solution is None:
        initial_solution = np.random.permutation(customers).astype(np.int64)
    
    # demands[1:] = np.random.randint(0, 2, NUM_NODES-1)
    demands[customers] = qs[customers-1]#np.ones(NUM_NODES-1)
    
    # Run optimized SA
    routes, costs, oq, assignment = simulated_annealing_vrp(
        D, demands, capacity, initial_solution, max_vehicles,
        Q = Q,
        **SA_configs,
    )
    
    # print(f"Total cost: {cost:.2f}")
    # print(f"Emissions: {cost - oq*3*np.amax(D):.2f}")
    if log:
        print(f"Emissions: {costs:.2f}")
        print(f"Best Oq: {oq}")
        print(f"Best r: {NUM_NODES -1 -oq}")
        print(f"Best r (assignment): {np.sum(assignment.astype(bool))}")
        print(f"Assignment: {assignment}")

        for i, route in enumerate(routes):
            if route.any():  # Skip empty routes
                print(f"Vehicle {i+1}: {route}")
    
    return costs, oq, routes, assignment

def SA_vrp2(distance_matrix, Q, qs, capacity, emissions_KM, 
           customers = None, log = False,
           initial_solution = None,
           SA_configs = dict(
              initial_temp=1_000,
              cooling_rate=0.995,
              max_iter=50_000, 
            ),
    ):
    
    # Problem parameters
    # em_factors = [.1, .1, .3, .3]
    # em_factors = [.1, .3]
    max_vehicles = len(emissions_KM)
    # Q = 100
    # excess_penalty = 10_000
    
    # # Generate multiple distance matrices
    # distance_matrices = np.random.randint(1, 100, (NUM_VEHICLES, NUM_NODES, NUM_NODES)).astype(np.float64)
    # for v in range(NUM_VEHICLES):
    #     np.fill_diagonal(distance_matrices[v], 0)
    #     distance_matrices[v] = (distance_matrices[v] + distance_matrices[v].T) / 2
    
    # distance_matrix = np.load('data/distance_matrix.npy').astype(np.float64)#[:100, :100]
    # scenarios = np.load('data/destinations_K100_100_test.npy').astype(np.int64)
    
    # s = [0] + list(scenarios[s_idx])
    NUM_NODES = len(distance_matrix)
    # mask = np.ix_(s, s)
    # distance_matrix = distance_matrix[mask]
    
    D = np.array([
        emissions_KM[v]*distance_matrix
        for v in range(max_vehicles)
    ]).astype(np.float64)
    
    
    # print(distance_matrices.shape)
    
    # # Generate demands
    # demands = np.zeros(NUM_NODES, dtype=np.int64)
    # # demands[1:] = np.random.randint(0, 2, NUM_NODES-1)
    # demands[1:] = np.ones(NUM_NODES-1)
    
    # # Generate sample data
    # np.random.seed(1917)
    # depot = np.array([[0, 0]])
    # customers = np.random.randint(0, 100, (20, 2))
    # coords = np.vstack((depot, customers))
    # emission = [.1, .3, .3]
    # # Problem setup
    # # n = coords.shape[0] - 1  # Exclude depot
    # # customers = np.arange(1, n + 1)
    # mat = compute_distance_matrix(coords)
    # D = np.array([
    #     mat*emission[v]
    #     for v in range(len(emission))
    # ])
    
    
    # demands = np.concatenate(([0], np.random.randint(1, 5, 20)))
    # Generate demands
    demands = np.zeros(NUM_NODES, dtype=np.int64)
    n = D.shape[1] - 1  # Exclude depot
    customers = np.arange(1, n + 1, dtype=np.int64) if customers is None else customers
    
    if initial_solution is None:
        initial_solution = distribute_random(customers, max_vehicles)
    else:
        initial_solution_numba = List()
        for l in initial_solution :
            initial_solution_numba.append(List(l))
        initial_solution = initial_solution_numba
        
    # demands[1:] = np.random.randint(0, 2, NUM_NODES-1)
    demands[customers] = qs[customers-1]#np.ones(NUM_NODES-1)
    
    # Run optimized SA
    routes, costs, oq, assignment = simulated_annealing_vrp2(
        D, demands, capacity, initial_solution, max_vehicles,
        Q = Q,
        **SA_configs,
    )
    
    # print(f"Total cost: {cost:.2f}")
    # print(f"Emissions: {cost - oq*3*np.amax(D):.2f}")
    if log:
        print(f"Emissions: {costs:.2f}")
        print(f"Best Oq: {oq}")
        print(f"Best r: {NUM_NODES -1 -oq}")
        print(f"Best r (assignment): {np.sum(assignment.astype(bool))}")
        print(f"Assignment: {assignment}")

        for i, route in enumerate(routes):
            if route.any():  # Skip empty routes
                print(f"Vehicle {i+1}: {route}")
    
    return costs, oq, routes, assignment
    
    
def multiple_SA_tsp(distance_matrix, Q, qs, capacity, emissions_KM, 
           initial_solutions,
           customers = None, log = False,
           SA_configs = dict(
              initial_temp=1_000,
              cooling_rate=0.995,
              max_iter=50_000, 
            ),
    ):
    
    
    def process(i, q):
        # a = dests[i_id][np.where(a_GTS == 0)].astype(int)
        
        np.random.seed(i)
        res = dict()
        cost, oq, routes, assignment = SA_vrp(
            distance_matrix, Q, qs[customers[i]-1], capacity, [emissions_KM[i]], 
            customers = customers[i], initial_solution = initial_solutions[i], log = log,
            SA_configs = SA_configs,
        )
        res['a'] = assignment
        res['routes'] = routes
        res['cost'] = cost
        res['oq'] = oq
        q.put((i, res))
        # print(f'DP {i} done')
        return
    
    n_vehicles = len(emissions_KM)
    cost = np.zeros(n_vehicles)
    oq = np.zeros(n_vehicles)
    assignment = np.zeros(len(distance_matrix)-1, np.int64)
    routes = np.zeros((n_vehicles, capacity+2), np.int64)
    # routes = [None for _ in range(n_vehicles)]
    # episode_rewards = np.zeros(n_sample)
    # actions = [[] for _ in range(n_sample)]
    
    q = mp.Manager().Queue()
    
    ps = []
    # pool = mp.Pool(processes=7)
    
    for i in range(len(customers)):
        ps.append(
            mp.Process(target = process, args = (i, q, ))
        )
        ps[-1].start()
    # p[4*i+3].start()
    
    for p in ps:
        p.join()
        
    while not q.empty():
        i, d = q.get()
        cost[i] = d["cost"]
        oq[i] = d["oq"]
        assignment += d["a"]
        routes[i] = d["routes"]
        
    # best_idx = np.argmax(episode_rewards)
        
    return cost, oq, routes, assignment

if __name__ == "__main__":
    # genetic()
    s_idx = 1
    Q = 50
    emissions_KM = [.1, .3]
    capacity = 20
    
    distance_matrix = np.load('data/distance_matrix.npy').astype(np.float64)#[:100, :100]
    scenarios = np.load('data/destinations_K100_100_test.npy').astype(np.int64)
    s = [0] + list(scenarios[s_idx])
    NUM_NODES = len(s)
    mask = np.ix_(s, s)
    distance_matrix = distance_matrix[mask]
    
    qs = np.ones(NUM_NODES-1)
    # mask = np.ix_(s, s)
    # distance_matrix = distance_matrix[mask]
    
    customers = np.arange(1, 15 + 1)
    
    SA_vrp(distance_matrix, Q, qs, capacity, emissions_KM, 
           customers = customers, log = True,
           SA_configs = dict(
              initial_temp=1_000,
              cooling_rate=0.995,
              max_iter=1_000, 
            ),
           )