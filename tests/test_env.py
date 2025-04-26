import unittest

import numpy as np

import sys
import os
# Add the DynamicQVRP directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs import DynamicQVRPEnv
from stable_baselines3.common.env_checker import check_env

def calculate_emissions(D, routes, emissions_KM):
    # calculates the emissions of all activated vehicles
    # routes : (n_vehicles, n_destinations)
    # Q : quota
    # costs_KM : (n_destinations, n_destinations)
    # emissions_KM : (n_destinations, n_destinations)   
    E = np.array([
        emissions_KM[v]*D
        for v, _ in enumerate(emissions_KM)
    ])
    
    emissions = 0.
    for v, route in enumerate(routes):
        for i in range(len(route)):
            emissions += E[v, route[i], route[i+1]]
            if route[i+1] == 0:
                break
    
    return emissions

class TestEnv(unittest.TestCase):

    def test_sb3(self):
        env = DynamicQVRPEnv()
        check_env(env)
        


class TestDynamicQVRPEnv(unittest.TestCase):

    def setUp(self):
        self.env = DynamicQVRPEnv()

    def test_reset(self):
        state, info = self.env.reset()
        self.assertIsNotNone(state, "Reset method should return the initial state")
        self.assertIsInstance(state, np.ndarray, "State should be a list")
        self.assertIsInstance(info, dict, "info should be a dictionary")

    def test_step(self):
        self.env.reset()
        action = 0  # Assuming action space includes 0
        next_state, reward, done, trunc, info = self.env.step(action)
        self.assertIsNotNone(next_state, "Step method should return the next state")
        self.assertIsInstance(next_state, np.ndarray, "Next state should be a list")
        self.assertIsInstance(reward, (int, float), "Reward should be a number")
        self.assertIsInstance(done, bool, "Done should be a boolean")
        self.assertIsInstance(trunc, bool, "Trunc should be a boolean")
        self.assertIsInstance(info, dict, "Info should be a dictionary")

    def test_action_space(self):
        self.assertTrue(hasattr(self.env, 'action_space'), "Environment should have an action space")
        self.assertIsNotNone(self.env.action_space, "Action space should not be None")

    def test_observation_space(self):
        self.assertTrue(hasattr(self.env, 'observation_space'), "Environment should have an observation space")
        self.assertIsNotNone(self.env.observation_space, "Observation space should not be None")
        
    def check_routes_and_assignment(self, routes, assignment, i = None):
        # checks if all the assigned destinations are in the route and vice versa
        rs = routes.flatten()
        n_dests_routes = len(rs[rs != 0])
        n_dests_assignment = len(assignment[assignment != 0])
        self.assertTrue(n_dests_routes == n_dests_assignment, 
            f"""
            The number of customers in routes and assignment do not match
            instance : {i}
            routes : {routes}
            customers : {np.sort(rs[rs != 0])}
            n_dests_routes : {n_dests_routes}
            assigned dests : {np.where(assignment!=0)[0]+1}
            n_dests_assignment : {n_dests_assignment}
            """
        )
        
        # checks if the destinations assignments are correct
        for v, route in enumerate(routes):
            customers = np.sort(route[route != 0])
            self.assertTrue((customers -1 == np.where(assignment==v+1)[0]).all(), 
                f"""
                \n
                Customers and assignment do not match
                customers : {customers}
                assignment : {assignment}
                """
            )
            
    def check_emissions_calculation(self, distance_matrix, emissions_KM, routes, info, Q):
        # checks if the emissions of all activated vehicles are calculated
        
        if 'emissions per vehicle' in info.keys():
            self.assertAlmostEqual(
                Q - info["remained_quota"],
                np.sum(info["emissions per vehicle"]), 
                msg=f"""
                The total emissions must match the sum of all emissions
                Q - info["remained_quota"] : {Q - info["remained_quota"]}
                emissions per vehicle : {info["emissions per vehicle"]}

                """
            )
            
            # checks the emissions calculation
            recalculated_emissions = calculate_emissions(
                distance_matrix, routes, emissions_KM
            )
            self.assertAlmostEqual(
                np.sum(recalculated_emissions),
                np.sum(info["emissions per vehicle"]), 
                msg=f"""
                The total emissions are not correctly calculated
                recalculated emissions : {recalculated_emissions}
                emissions in info : {info["emissions per vehicle"]}

                """
            )
            
            vehicle_activation = np.sum(routes, axis=1).astype(bool)
            emissions_calculated = info['emissions per vehicle'].astype(bool)

            self.assertTrue((vehicle_activation == emissions_calculated).all(), 
                f"""
                \n
                Emissions are not calculated properly
                activated vehicles : {vehicle_activation}
                emissions : {info['emissions per vehicle']}
                routes : {routes}
                """
            )

        
        
    def test_observations(self):
        for _ in range(len(self.env.all_dests)):
            state, _ = self.env.reset()
            self.assertTrue(self.env.observation_space.contains(state), 
                            f"The obs must be in observation space, obs : {state}")
            while True:
                action = self.env.action_space.sample()
                state, _, done, trun, info = self.env.step(action)
                self.assertTrue(self.env.observation_space.contains(state), f"The obs must be in observation space, obs : {state}")
                self.assertTrue(info["remained_quota"] + 1e-4 >= 0, 'The quota must be respected')
                self.check_emissions_calculation(
                    self.env.distance_matrix, self.env.emissions_KM,
                    self.env.routes, info, self.env.Q
                )
                self.check_routes_and_assignment(self.env.routes, self.env.assignment)
                if done or trun:
                    break
                
    def test_quantities_online(self):
        env = DynamicQVRPEnv(DoD=1, different_quantities=True)
        for _ in range(len(env.all_dests)):
            state, _ = env.reset()
            self.assertTrue(env.observation_space.contains(state), 
                            f"The obs must be in observation space, obs : {state}")
            # self.assertTrue(env.reset(), f"The obs must be in observation space, obs : {state}")
            qs  = env.quantities.copy()
            while True:
                action = env.action_space.sample()
                state, _, done, trun, info = env.step(action)
                self.assertTrue(env.observation_space.contains(state), f"The obs must be in observation space, obs : {state}")
                self.assertTrue(info["remained_quota"] + 1e-4 >= 0, 'The quota must be respected')
                self.check_emissions_calculation(
                    env.distance_matrix, env.emissions_KM,
                    env.routes, info, env.Q
                )
                self.assertTrue((env.quantities == qs).all(), "Quantities should not change")
                if done or trun:
                    break
                
    def test_quantities_online_VRP(self):
        env = DynamicQVRPEnv(DoD=1., different_quantities=True, costs_KM=[1, 1], emissions_KM=[.1, .3])
        for i in range(len(env.all_dests)):
            state, _ = env.reset(i)
            self.assertTrue(env.observation_space.contains(state), 
                            f"The obs must be in observation space, obs : {state}")
            # self.assertTrue(env.reset(i), f"The obs must be in observation space, obs : {state}")
            qs  = env.quantities.copy()
            while True:
                action = env.action_space.sample()
                state, _, done, trun, info = env.step(action)
                self.assertTrue(env.observation_space.contains(state), f"The obs must be in observation space, obs : {state}")
                self.assertTrue(info["remained_quota"] + 1e-4 >= 0, 'The quota must be respected')
                # print(env.quantities)
                self.check_emissions_calculation(
                    env.distance_matrix, env.emissions_KM,
                    env.routes, info, env.Q
                )
                self.check_routes_and_assignment(env.routes, env.assignment, i)
                self.assertTrue((env.quantities == qs).all(), "Quantities should not change")
                if done or trun:
                    break
                
    def test_quantities_online_VRP_wReOpt(self):
        env = DynamicQVRPEnv(
            DoD=1., different_quantities=True, costs_KM=[1, 1], emissions_KM=[.1, .3],
            re_optimization=True,)
        for i in range(len(env.all_dests)):
            state, _ = env.reset(i)
            self.assertTrue(env.observation_space.contains(state), 
                            f"The obs must be in observation space, obs : {state}")
            qs  = env.quantities.copy()
            while True:
                action = env.action_space.sample()
                state, _, done, trun, info = env.step(action)
                self.assertTrue(env.observation_space.contains(state), f"The obs must be in observation space, obs : {state}")
                self.assertTrue(info["remained_quota"] + 1e-4 >= 0, 'The quota must be respected')
                self.check_emissions_calculation(
                    env.distance_matrix, env.emissions_KM,
                    env.routes, info, env.Q
                )
                self.check_routes_and_assignment(env.routes, env.assignment, i)
                self.assertTrue((env.quantities == qs).all(), "Quantities should not change")
                if done or trun:
                    break
                
    def test_quantities(self):
        env = DynamicQVRPEnv(different_quantities=True)
        for _ in range(len(env.all_dests)):
            state, _ = env.reset()
            self.assertTrue(env.observation_space.contains(state), 
                            f"The obs must be in observation space, obs : {state}")
            
    def test_quantities_VRP(self):
        env = DynamicQVRPEnv(different_quantities=True, costs_KM=[1, 1], emissions_KM=[.1, .3])
        for _ in range(len(env.all_dests)):
            state, _ = env.reset()
            self.assertTrue(env.observation_space.contains(state), 
                            f"The obs must be in observation space, obs : {state}")
            
    def test_vehicle_assignment(self):
        env = DynamicQVRPEnv(vehicle_assignment=True, costs_KM=[1, 1], emissions_KM=[.1, .3])
        for _ in range(len(env.all_dests)):
            state, _ = env.reset()
            self.assertTrue(env.observation_space.contains(state), 
                            f"The obs must be in observation space, obs : {state}")
            while True:
                action = env.action_space.sample()
                state, _, done, trun, info = env.step(action)
                self.assertTrue(True, 'The environment should not raise an error')
                self.assertTrue(info["remained_quota"] + 1e-4 >= 0, 'The quota must be respected')
                self.check_emissions_calculation(
                    env.distance_matrix, env.emissions_KM,
                    env.routes, info, env.Q
                )
                self.check_routes_and_assignment(env.routes, env.assignment)
                
                if done or trun:
                    break
                self.assertTrue(
                    env.observation_space.contains(state), 
                    f"The obs must be in observation space, obs : {state}"
                )
                
    def test_cluster_scenario(self):
        env = DynamicQVRPEnv(
            vehicle_assignment=True, costs_KM=[1, 1], emissions_KM=[.1, .3],
            cluster_scenario=True
            )
        for _ in range(len(env.all_dests)):
            state, _ = env.reset()
            self.assertTrue(env.observation_space.contains(state), 
                            f"The obs must be in observation space, obs : {state}")
            while True:
                action = env.action_space.sample()
                state, _, done, trun, info = env.step(action)
                if done or trun:
                    break
                self.assertTrue(True, 'The environment should not raise an error')
                self.assertTrue(
                    env.observation_space.contains(state), 
                    f"""The obs must be in observation space
                    obs : {state}
                    """
                )
                self.assertTrue(info["remained_quota"] + 1e-4 >= 0, 'The quota must be respected')
                self.check_emissions_calculation(
                    env.distance_matrix, env.emissions_KM,
                    env.routes, info, env.Q
                )
                self.check_routes_and_assignment(env.routes, env.assignment)
                
                
    def test_uniform_scenario(self):
        env = DynamicQVRPEnv(
            horizon=100,
            vehicle_assignment=True, costs_KM=[1, 1], emissions_KM=[.1, .3],
            uniform_scenario=True
            )
        for _ in range(len(env.all_dests)):
            state, _ = env.reset()
            self.assertTrue(env.observation_space.contains(state), 
                            f"The obs must be in observation space, obs : {state}")
            while True:
                action = env.action_space.sample()
                state, _, done, trun, info = env.step(action)
                if done or trun:
                    break
                self.assertTrue(True, 'The environment should not raise an error')
                self.assertTrue(
                    env.observation_space.contains(state), 
                    f"""The obs must be in observation space
                    obs : {state}
                    """
                )
                self.assertTrue(info["remained_quota"] + 1e-4 >= 0, 'The quota must be respected')
                self.check_emissions_calculation(
                    env.distance_matrix, env.emissions_KM,
                    env.routes, info, env.Q
                )
                self.check_routes_and_assignment(env.routes, env.assignment)
                
    def test_diffrent_DoDs(self):
        
        for DoD in [1., .75, .5, .25, 0]:
            env = DynamicQVRPEnv(
                DoD=DoD,
                costs_KM=[1, 1], emissions_KM=[.1, .3],
                )
            # state, _ = self.env.reset()
            s_idx = np.random.randint(len(env.all_dests))
            state, _ = env.reset(s_idx)
            self.assertTrue(env.observation_space.contains(state), 
                            f"The obs must be in observation space, obs : {state}")
            # self.assertTrue(env.reset(s_idx), f"The obs must be in observation space, obs : {state}")
            while True:
                action = env.action_space.sample()
                state, _, done, trun, *_ = env.step(action)
                self.assertTrue(True, 'The environment should not raise an error')
                self.assertTrue(env.observation_space.contains(state), f"The obs must be in observation space, obs : {state}")
                if done or trun:
                    break

if __name__ == '__main__':
    unittest.main()