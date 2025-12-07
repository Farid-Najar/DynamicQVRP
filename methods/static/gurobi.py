import numpy as np
from gurobipy import Model, GRB, quicksum

def top_gurobi(num_agents, nodes, prizes, depot, max_length, timeout=0):
    num_nodes = len(nodes)
    nodes = np.array(nodes)
    depot = np.array(depot)
    
    # Calculate distance matrix
    all_nodes = np.vstack([depot, nodes])
    dist_matrix = np.linalg.norm(all_nodes[:, None] - all_nodes, axis=2)

    # Create Gurobi model
    model = Model('Team Orienteering Problem')
    model.Params.outputFlag = False
    
    # Decision variables
    x = model.addVars(num_agents, num_nodes + 1, num_nodes + 1, vtype=GRB.BINARY, name='x')
    u = model.addVars(num_agents, num_nodes + 1, vtype=GRB.CONTINUOUS, name='u')
    
    # Objective function: maximize total prize collected
    model.setObjective(
        quicksum(prizes[j-1] * x[i, j, k] for i in range(num_agents) for j in range(1, num_nodes + 1) for k in range(num_nodes + 1)),
        GRB.MAXIMIZE
    )
    
    # Constraints
    for i in range(num_agents):
        
        # Start from the depot
        model.addConstr(quicksum(x[i, 0, j] for j in range(1, num_nodes + 1)) == 1)
        
        # End at the depot
        model.addConstr(quicksum(x[i, j, 0] for j in range(1, num_nodes + 1)) == 1)
        
        # Do not go from node j to node j
        model.addConstr(quicksum(x[i, j, j] for j in range(1, num_nodes + 1)) == 0)
        
        # Flow constraints
        for j in range(1, num_nodes + 1):
            model.addConstr(quicksum(x[i, k, j] for k in range(num_nodes + 1)) == quicksum(x[i, j, k] for k in range(num_nodes + 1)))
        
        # Sub-tour elimination constraints
        for j in range(1, num_nodes + 1):
            for k in range(1, num_nodes + 1):
                if j != k:
                    model.addConstr(u[i, j] - u[i, k] + num_nodes * x[i, j, k] <= num_nodes - 1)

        # Distance constraint
        model.addConstr(quicksum(dist_matrix[j, k] * x[i, j, k] for j in range(num_nodes + 1) for k in range(num_nodes + 1)) <= max_length[i])
    
    # Add constraint to ensure that each non-depot node is visited by at most one agent
    for j in range(1, num_nodes + 1):
        model.addConstr(quicksum(x[i, k, j] for i in range(num_agents) for k in range(num_nodes + 1)) <= 1)
    
    # Optimize the model
    if timeout:
        model.Params.timeLimit = timeout
    model.Params.lazyConstraints = 1
    model.Params.threads = 0
    model.optimize()
    
    # Get the results
    results = []
    for i in range(num_agents):
        path = []
        current = 0  # Start from depot
        visited = set()  # Track visited nodes
        while True:
            path.append(current)
            visited.add(current)
            
            # Check if model was successful
            if not hasattr(x[i, current, k], "X"):
                print(f"[Warning] Model did not solve successfully")
                return [], model.Runtime
            
            # Find the next node
            next_node = [k for k in range(num_nodes + 1) if x[i, current, k].X > 0.5 and k not in visited]
            
            # No more unvisited nodes
            if not next_node:
                break
            current = next_node[0]
        results.append(path)
    
    return results, model.Runtime



def top_gurobi(num_vehicles, cost_matrices, q, Q, Cap, timeout=0):
    """
    Solves the Team Orienteering Problem with:
    - Heterogeneous vehicles (each with own cost matrix)
    - Global time budget constraint
    - Per-vehicle capacity limits
    
    Parameters:
    num_vehicles: int - Number of vehicles
    cost_matrices: list[np.ndarray] - List of (n+1)x(n+1) cost matrices for each vehicle
    q: np.ndarray - Reward values for customer nodes (length = n)
    Q: float - Global time budget (sum of all route durations)
    Cap: int - Maximum number of customer nodes per vehicle
    timeout: int - Solver time limit in seconds (0 = no limit)
    """
    num_nodes = len(q)  # Customer nodes only (depot excluded)
    n_total = num_nodes + 1  # Including depot (index 0)
    
    # Create Gurobi model
    model = Model('Heterogeneous TOP with Global Budget')
    model.Params.outputFlag = False
    
    # Decision variables
    x = model.addVars(num_vehicles, n_total, n_total, vtype=GRB.BINARY, name='x')
    u = model.addVars(num_vehicles, n_total, vtype=GRB.CONTINUOUS, name='u')
    
    # Objective: maximize total collected prizes
    model.setObjective(
        quicksum(
            q[j-1] * x[i, j, k]
            for i in range(num_vehicles)
            for j in range(1, n_total)    # Customer nodes only
            for k in range(n_total)        # All nodes
            if j != k
        ),
        GRB.MAXIMIZE
    )
    
    # Constraints
    for i in range(num_vehicles):
        # Route structure constraints
        model.addConstr(quicksum(x[i, 0, j] for j in range(1, n_total)) == 1, 
                       f"start_depot_{i}")
        model.addConstr(quicksum(x[i, j, 0] for j in range(1, n_total)) == 1,
                       f"end_depot_{i}")
        model.addConstr(quicksum(x[i, j, j] for j in range(n_total)) == 0,
                       f"no_self_loops_{i}")
        
        # Flow conservation for customer nodes
        for j in range(1, n_total):
            model.addConstr(
                quicksum(x[i, k, j] for k in range(n_total)) == 
                quicksum(x[i, j, k] for k in range(n_total)),
                f"flow_cons_{i}_{j}"
            )
        
        # Capacity constraint: max customer nodes per vehicle
        model.addConstr(
            quicksum(x[i, j, k] 
                    for j in range(1, n_total) 
                    for k in range(n_total) 
                    if j != k) <= Cap,
            f"capacity_{i}"
        )
        
        # MTZ subtour elimination
        for j in range(1, n_total):
            for k in range(1, n_total):
                if j != k:
                    model.addConstr(
                        u[i, j] - u[i, k] + n_total * x[i, j, k] <= n_total - 1,
                        f"mtz_{i}_{j}_{k}"
                    )
    
    # Global time budget constraint
    total_time = quicksum(
        cost_matrices[i][j, k] * x[i, j, k]
        for i in range(num_vehicles)
        for j in range(n_total)
        for k in range(n_total)
        if j != k
    )
    model.addConstr(total_time <= Q, "global_time_budget")
    
    # Each customer visited at most once
    for j in range(1, n_total):
        model.addConstr(
            quicksum(x[i, k, j] 
                    for i in range(num_vehicles) 
                    for k in range(n_total)) <= 1,
            f"single_visit_{j}"
        )
    
    # Solver configuration
    if timeout > 0:
        model.Params.timeLimit = timeout
    model.Params.threads = 0  # Use default number of threads
    
    # Optimize
    model.optimize()
    
    # Extract solution
    if model.status != GRB.OPTIMAL and model.status != GRB.TIME_LIMIT:
        return [], model.Runtime
    
    routes = []
    total_e = 0.
    for i in range(num_vehicles):
        route = [0]  # Start at depot
        current = 0
        visited = {0}
        while True:
            next_nodes = [
                k for k in range(n_total)
                if k not in visited 
                and x[i, current, k].X > 0.5
            ]
            if not next_nodes:
                break
            next_node = next_nodes[0]
            route.append(next_node)
            visited.add(next_node)
            total_e += cost_matrices[i][current, next_node]
            current = next_node
        
        # Ensure route ends at depot if not already there
        if route[-1] != 0:
            route.append(0)
        routes.append(route)
    
    return routes, model.Runtime, total_e