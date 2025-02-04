import numpy as np
import numba as nb
from numba import njit
from numba.typed import List
import random

# Genetic Algorithm Parameters
POPULATION_SIZE = 500
GENERATIONS = 2_000
MUTATION_RATE = 0.1
ELITISM_COUNT = 7
TOURNAMENT_SIZE = 5
VEHICLE_PENALTY_FACTOR = 0  # Penalty multiplier for exceeding vehicle limit
# EXCESS_PENALTY = 10_000

# njit = lambda x, *args : x

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
            if v+1 >= MAX_VEHICLES:
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
                distance_matrices[v, current_route[-1], customer] +
                distance_matrices[v, customer, self.depot] )> self.Q:
                oq += demand
                vehicle_assignments[customer] = -1
                # continue

            elif current_load + demand > self.vehicle_capacity:
                # Finalize current route
                total_distance += distance_matrices[v, current_route[-1], self.depot]
                current_route.append(self.depot)
                routes.append(np.array(current_route, dtype=np.int64))
                
                # route_length = 1
                current_load = demand
                if v+1 >= MAX_VEHICLES:
                    oq += np.sum(self.demands[individual[i+1:]])
                    print(individual[i+1:])
                    break
                
                v+=1
                # Start new route
                current_route = [self.depot, customer]
                total_distance += distance_matrices[v, self.depot, customer]
                vehicle_assignments[customer] = v
            else:
                total_distance += distance_matrices[v, current_route[-1], customer]
                # route_length += 1
                current_route.append(customer)
                current_load += demand

                vehicle_assignments[customer] = v
          
        
        total_distance += distance_matrices[v, current_route[-1], self.depot]
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
if __name__ == "__main__":
    # Problem parameters
    s_idx = 0
    NUM_NODES = 51
    VEHICLE_CAPACITY = 20
    MAX_VEHICLES = 2
    NUM_VEHICLES = 2  # Total available vehicle types
    # Q = 100
    # excess_penalty = 10_000
    
    # # Generate multiple distance matrices
    # distance_matrices = np.random.randint(1, 100, (NUM_VEHICLES, NUM_NODES, NUM_NODES)).astype(np.float64)
    # for v in range(NUM_VEHICLES):
    #     np.fill_diagonal(distance_matrices[v], 0)
    #     distance_matrices[v] = (distance_matrices[v] + distance_matrices[v].T) / 2
    
    distance_matrix = np.load('data/distance_matrix.npy').astype(np.float64)#[:100, :100]
    scenarios = np.load('data/destinations_K50_100_test.npy').astype(np.int64)
    s = [0] + list(scenarios[s_idx])
    mask = np.ix_(s, s)
    distance_matrix = distance_matrix[mask]
    
    em_factors = [.1, .3]
    distance_matrices = np.array([
        em_factors[v]*distance_matrix
        for v in range(NUM_VEHICLES)
    ]).astype(np.float64)
    
    # print(distance_matrices.shape)
    
    # Generate demands
    demands = np.zeros(NUM_NODES, dtype=np.int64)
    # demands[1:] = np.random.randint(0, 2, NUM_NODES-1)
    demands[1:] = np.ones(NUM_NODES-1)
    
    # Initialize and run solver
    solver = MultiVehicleVRPSolver(
        distance_matrices=distance_matrices,
        demands=demands,
        vehicle_capacity=VEHICLE_CAPACITY,
        max_vehicles=MAX_VEHICLES
    )
    
    best_solution, best_score = solver.solve()
    routes, vehicles, oq = solver.decode_solution(best_solution)
    
    print("\nBest solution:")
    print(f"Total score: {best_score:.2f}")
    print(f"Vehicles assignments: {vehicles+1}")
    lengths = 0
    for i, route in enumerate(routes):
        lengths += len(route) -2 
        print(f"Route {i+1} (Vehicle {i+1}): {route.tolist()}")
    # print(f"Total omitted: {oq:.2f}")
    print(f"Total rewards: {NUM_NODES-1 - oq:.2f}")
    assert lengths == NUM_NODES-1 - oq
    # print(f"lenghts: {lenghts:.2f}")