from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from sortedcontainers import SortedList
import numpy as np

from typing import List

def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f'Objective: {solution.ObjectiveValue()}')
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                 route_load)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        plan_output += 'Load of the route: {}\n'.format(route_load)
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print('Total distance of all routes: {}m'.format(total_distance))
    print('Total load of all routes: {}'.format(total_load))
    
def solve(data, time_budget):
    """Returns the CVRP solution and routing"""
    
    # # Sets a time limit of 10 seconds.
    # search_parameters.time_limit.seconds = 10
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance']),
                                           data['num_vehicles'], data['depot'])
    # print('n vehicles : ', data['num_vehicles'])
    # print('n vehicles : ', data['distance_matrix'])
    
    
    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)
    
    # transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    
    # print('test', distance_callback(5,7))
    
    # Define cost of each arc.
    def distance_callback(from_index, to_index, vehicle_id):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['cost_matrix'][vehicle_id, from_node, to_node]
    
    # routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    for m in range(data['num_vehicles']):
            # Create and register a transit callback.
        
        transit_callback_index = routing.RegisterTransitCallback(
            lambda from_idx, to_idx : distance_callback(
                from_idx, to_idx, m)
        )
        routing.SetArcCostEvaluatorOfVehicle(transit_callback_index, m)
        
    # print('sdsdsd', distance_callback(3,2, 0))
    # print('sdsdsd', distance_callback(3,2, 1))
    # print('sdsdsd', distance_callback(3,2, 2))
    
    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback
    )
    
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity'
    )
    
    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.FromSeconds(time_budget)
    # search_parameters.use_depth_first_search = True
    
    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    # Print solution on console.
    #if solution:
    #    print_solution(data, manager, routing, solution)
    
    return solution, routing, manager

class Transporter:
    def __init__(self, 
                 distance_matrix,
                 cost_matrix,
                #  emission_per_unit = None,
                 time_matrix = None,
                 omission_cost = 100, 
                 transporter_hub : int = 85, 
                 max_capacity : int|List[int] = 15,
                 num_vehicles :int = 1,
                 ):
        
        self.data = dict()
        self.orders = [] # List of nodes to visit
        
        if type(max_capacity) == List:
            self.data['vehicle_capacities'] = max_capacity
        elif type(max_capacity) == int:
            self.data['vehicle_capacities'] = [max_capacity for _ in range(num_vehicles)]
        else:
            raise("max_capacity must be an integer or a list of integers !")
        
        self.max_capacity = 100
        self.capacity = 0
        self.omission_cost = omission_cost
        self.distance_matrix = distance_matrix
        self.cost_matrix = cost_matrix
        self.transporter_hub = transporter_hub
        self.nodes = SortedList()

        # if emission_per_unit is None:
        #     self.data['emission_per_unit'] = np.ones(num_vehicles, dtype=int)
        # else:
        #     self.data['emission_per_unit'] = cost_per_unit.copy()
        
        if time_matrix is None:
            self.time_matrix = distance_matrix/40 #In cities, the average speed is 40 km/h
        else:
            self.time_matrix = time_matrix
        
        self.last_cost = 0
        self.cost_history = [0]
        self.data['demands'] = []
        self.data['num_vehicles'] = num_vehicles
        self.data['depot'] = 0
        
    def reset(self):
        self.capacity = 0
        self.orders.clear()
        
    def new_order(self, node, quantity):
        self.nodes.add(node)
        idx = self.nodes.index(node)
        self.orders.insert(idx, quantity)
        self.cost_history.append(self.last_cost)
        
    

    def compute_cost(self, nodes, quantities, time_budget = 2):
        """Solve the CVRP problem and computes the total cost and time per vehicle"""
        # Instantiate the data problem.
        data = self.data
        
        data['demands'] = np.zeros(len(quantities)+1, dtype=int)
        
        data['demands'][1:] = np.array(quantities, dtype=int)
        
        l = [self.transporter_hub] + list(nodes)
        x, y = np.ix_(l, l)
        data['cost_matrix'] = self.cost_matrix[:, x, y]
        data['distance'] = self.distance_matrix[x, y]
        
        # print(data['distance_matrix'])
        
        solution, routing, manager = solve(data, time_budget)
        
        # print_solution(data, manager, routing, solution)
    
        # If no solution; we penalize all packages (it's a rare event)
        if not solution:
            return self.omission_cost*len(nodes), 0, 'solution not found !'
        
        # route_distance = np.zeros(data['num_vehicles'])
        # route_time = np.zeros(data['num_vehicles'])
        # text = 'Routes :'
        
        routes = [[] for _ in range(data['num_vehicles'])]
        
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            # text += '\nvehicle ' + str(vehicle_id) + '\n'
            # text += '0'
            routes[vehicle_id].append(0)
                
            while not routing.IsEnd(index):
                previous_index = index
                # print(index)
                index = solution.Value(routing.NextVar(index))
                # from_node = manager.IndexToNode(previous_index)
                to_node = manager.IndexToNode(index)
                
                # text += ' -> ' + str(to_node)
                routes[vehicle_id].append(to_node)
                
                # route_distance[vehicle_id] += data['distance'][from_node, to_node]
                # routing.GetArcCostForVehicle(
                #     previous_index, index, vehicle_id
                # )
                # route_time[vehicle_id] += self.time_matrix[from_node, to_node]
                # print(route_distance)
                
        return routes
         

if __name__ == '__main__':
    import networkx as nx
    size = 15
    G = nx.grid_2d_graph(size, size)
    emissions_KM = [0, .15, .3, .3]
    costs_KM = [4, 4, 4, 4]
    CO2_penalty = 1_000
    distance_matrix = nx.floyd_warshall_numpy(G)
    cost_matrix = np.array([
            (costs_KM[m] + CO2_penalty*emissions_KM[m])*distance_matrix
            for m in range(len(costs_KM))
    ], dtype=int)
    T1 = Transporter(distance_matrix, cost_matrix, num_vehicles=4)
    # c = T1.compute_marginal_cost(1, 5)
    # print(c)
    # T1.new_order(1, 5)
    # c = T1.compute_marginal_cost(3, 5)
    # print(c)
    
    nodes = np.random.choice(len(G.nodes), size=size, replace=False)
    quantities = np.ones(size, dtype=int)
    route_distance, route_time, text = T1.compute_cost(nodes, quantities)
    print(text)
    print('total cost : ', np.sum(route_distance))
