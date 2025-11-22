import numpy as np
import random
from typing import List, Tuple, Dict

class ACO_TOP:
    def __init__(
        self,
        cost_matrices: List[np.ndarray],   # [K x (n+1) x (n+1)]
        rewards: np.ndarray,               # [n+1], rewards[0] = 0 (depot)
        Q: float,                          # global time budget
        Cap: int,                          # max number of customer nodes per vehicle
        num_ants: int = 10,
        max_iter: int = 200,
        rho: float = 0.1,                  # pheromone evaporation
        alpha: float = 1.0,                # pheromone influence
        beta: float = 2.0,                 # heuristic influence
        seed: int = 42
    ):
        np.random.seed(seed)
        random.seed(seed)
        
        self.K = len(cost_matrices)                # number of vehicles
        self.n_plus_1 = cost_matrices[0].shape[0]  # total nodes including depot (0)
        self.n = self.n_plus_1 - 1                 # customer nodes: 1..n
        self.cost_matrices = [np.array(C, dtype=float) for C in cost_matrices]
        self.rewards = np.array(rewards, dtype=float)
        self.Q = Q
        self.Cap = Cap
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.rho = rho
        self.alpha = alpha
        self.beta = beta

        # Initialize pheromone: one matrix per vehicle (only for customer nodes)
        # Pheromone[k][i][j] = tau for vehicle k from i to j (i,j ∈ [0..n])
        tau0 = 1.0 / (self.n * np.mean([np.mean(C) for C in self.cost_matrices]))
        self.pheromone = [
            np.full((self.n_plus_1, self.n_plus_1), tau0) for _ in range(self.K)
        ]
        
        # Heuristic info: eta[i][j] = q_j / c_ij (but avoid div by zero)
        # We'll compute per vehicle during construction due to heterogeneous costs
        self.best_solution = None
        self.best_reward = -1.0

    def construct_solution(self) -> Tuple[List[List[int]], float, float]:
        """Construct a feasible solution using ants (one ant per vehicle implicitly via greedy randomized)."""
        unvisited = set(range(1, self.n_plus_1))
        routes = [[] for _ in range(self.K)]
        total_time = 0.0
        total_reward = 0.0
        visited = set()

        # We'll simulate K ants sequentially (one per vehicle)
        for k in range(self.K):
            route = [0]  # start at depot
            current = 0
            time_used = 0.0
            stops = 0
            local_visited = []

            while stops < self.Cap and unvisited:
                # Compute candidate nodes: unvisited and feasible (return to depot possible)
                candidates = []
                probs = []
                Ck = self.cost_matrices[k]

                for j in unvisited:
                    # Check if we can go: current -> j -> 0 within remaining global time?
                    time_to_j = Ck[current, j]
                    time_back = Ck[j, 0]
                    extra_time = time_to_j + time_back
                    if total_time + extra_time <= self.Q:
                        candidates.append(j)
                        # Heuristic desirability
                        c_ij = Ck[current, j]
                        eta = self.rewards[j] / (c_ij + 1e-6)
                        tau = self.pheromone[k][current, j]
                        prob = (tau ** self.alpha) * (eta ** self.beta)
                        probs.append(prob)
                    # else: not feasible

                if not candidates:
                    break

                # Normalize probabilities
                probs = np.array(probs)
                probs /= probs.sum()
                next_node = np.random.choice(candidates, p=probs)

                # Move
                time_used += Ck[current, next_node]
                total_time += Ck[current, next_node]
                route.append(next_node)
                local_visited.append(next_node)
                visited.add(next_node)
                unvisited.remove(next_node)
                stops += self.rewards[current]
                current = next_node

            # Return to depot
            if len(route) > 1:
                return_time = self.cost_matrices[k][current, 0]
                total_time += return_time
                route.append(0)

            routes[k] = route

        # Collect total reward from visited nodes (avoid double count)
        total_reward = self.rewards[list(visited)].sum() if visited else 0.0

        # Reconstruct unvisited for caller if needed (not used here)
        return routes, total_reward, total_time

    def global_update(self, routes: List[List[int]], reward: float):
        """Update pheromone based on best-so-far solution."""
        # Evaporate
        for k in range(self.K):
            self.pheromone[k] *= (1 - self.rho)

        # Deposit pheromone only if this is the best or best-so-far
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_solution = [r.copy() for r in routes]

            # Deposit pheromone on used arcs
            for k, route in enumerate(routes):
                if len(route) < 3:
                    continue  # no customer visited
                delta_tau = reward / len(route)  # or just reward
                for i in range(len(route) - 1):
                    a, b = route[i], route[i+1]
                    self.pheromone[k][a, b] += delta_tau
                    # Optional: symmetric update
                    # self.pheromone[k][b, a] += delta_tau

    def solve(self) -> Tuple[List[List[int]], float, float]:
        for it in range(self.max_iter):
            best_ant_reward = -1
            best_ant_sol = None
            best_ant_time = 0.0

            for _ in range(self.num_ants):
                routes, reward, time_used = self.construct_solution()
                if time_used <= self.Q + 1e-6 and reward > best_ant_reward:
                    best_ant_reward = reward
                    best_ant_sol = routes
                    best_ant_time = time_used

            if best_ant_sol is not None:
                self.global_update(best_ant_sol, best_ant_reward)

            # Optional: print progress
            # print(f"Iter {it}: Best reward = {self.best_reward:.2f}")

        # Final validation
        if self.best_solution is None:
            return [[] for _ in range(self.K)], 0.0, 0.0

        # Recompute final time & reward to be safe
        total_time = 0.0
        visited = set()
        for k, route in enumerate(self.best_solution):
            if len(route) < 2:
                continue
            for i in range(len(route) - 1):
                total_time += self.cost_matrices[k][route[i], route[i+1]]
            for node in route[1:-1]:  # exclude depots
                visited.add(node)
        final_reward = self.rewards[list(visited)].sum() if visited else 0.0

        return self.best_solution, final_reward, total_time


# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    from envs import StaticQVRPEnv
    

    env = StaticQVRPEnv(
            obs_mode='action',
            is_0_allowed=False,
            # DQVRP Env kwargs
            horizon = 100,
            Q = 50,
            vehicle_assignment = True,
            test = True,
            vehicle_capacity = 25,
            re_optimization = False,
            emissions_KM = [.1, .1, .3, .3],
        )
        
    # Test
    _ = env.reset(0)
    # Example: 2 vehicles, 4 nodes (0=depot, 1-3=customers)
    n_customers = 100
    K = 4
    # Cost matrices: shape (4,4) for each vehicle
    
    cost_matrices = env._env.cost_matrix

    rewards = np.zeros(n_customers + 1)  # q[0] = 0
    rewards[1:] = env._env.quantities
    Q = env._env.Q   # global total time budget
    Cap = env._env.max_capacity    # max 2 stops per vehicle

    aco = ACO_TOP(
        cost_matrices=cost_matrices,
        rewards=rewards,
        Q=Q,
        Cap=Cap,
        num_ants=50,
        max_iter=100,
        rho=0.1,
        alpha=1.0,
        beta=2.0,
        seed=42
    )

    routes, reward, total_time = aco.solve()

    print("Best routes:")
    for k, r in enumerate(routes):
        print(f"Vehicle {k}: {r}")
    print(f"Total reward: {reward:.2f}")
    print(f"Total time used: {total_time:.2f} (budget: {Q})")