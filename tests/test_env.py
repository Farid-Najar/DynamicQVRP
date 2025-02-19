import unittest

import numpy as np

import sys
import os
# Add the DynamicQVRP directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs import DynamicQVRPEnv
from stable_baselines3.common.env_checker import check_env

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
        
    def check_routes_and_assignment(self, routes, assignment):
        # checks if all the assigned destinations are in the route and vice versa
        rs = routes.flatten()
        n_dests_routes = len(rs[rs != 0])
        n_dests_assignment = len(assignment[assignment != 0])
        self.assertTrue(n_dests_routes == n_dests_assignment, 
            "The number of customers in routes and assignment do not match"
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
            
    def check_emissions_calculation(self, routes, info):
        # checks if the emissions of all activated vehicles are calculated
        if 'emissions per vehicle' in info.keys():
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
            self.assertTrue(self.env.observation_space.contains(state), "The obs must be in observation space")
            while True:
                action = self.env.action_space.sample()
                state, _, done, _, info = self.env.step(action)
                self.assertTrue(self.env.observation_space.contains(state), "The obs must be in observation space")
                self.assertTrue(info["remained_quota"] + 1e-4 >= 0, 'The quota must be respected')
                self.check_emissions_calculation(self.env.routes, info)
                self.check_routes_and_assignment(self.env.routes, self.env.assignment)
                if done:
                    break
                
    def test_quantities_online(self):
        env = DynamicQVRPEnv(DoD=1, different_quantities=True)
        for _ in range(len(env.all_dests)):
            # state, _ = self.env.reset()
            self.assertTrue(env.reset(), "The obs must be in observation space")
            qs  = env.quantities.copy()
            while True:
                action = env.action_space.sample()
                state, _, done, _, info = env.step(action)
                self.assertTrue(env.observation_space.contains(state), "The obs must be in observation space")
                self.assertTrue(info["remained_quota"] + 1e-4 >= 0, 'The quota must be respected')
                self.check_emissions_calculation(env.routes, info)
                self.assertTrue((env.quantities == qs).all(), "Quantities should not change")
                if done:
                    break
                
    def test_quantities_online_VRP(self):
        env = DynamicQVRPEnv(DoD=1, different_quantities=True, costs_KM=[1, 1], emissions_KM=[.1, .3])
        for _ in range(len(env.all_dests)):
            # state, _ = self.env.reset()
            self.assertTrue(env.reset(), "The obs must be in observation space")
            qs  = env.quantities.copy()
            while True:
                action = env.action_space.sample()
                state, _, done, _, info = env.step(action)
                self.assertTrue(env.observation_space.contains(state), "The obs must be in observation space")
                self.assertTrue(info["remained_quota"] + 1e-4 >= 0, 'The quota must be respected')
                self.check_emissions_calculation(env.routes, info)
                self.check_routes_and_assignment(env.routes, env.assignment)
                self.assertTrue((env.quantities == qs).all(), "Quantities should not change")
                if done:
                    break
                
    def test_quantities(self):
        env = DynamicQVRPEnv(different_quantities=True)
        for _ in range(len(env.all_dests)):
            # state, _ = self.env.reset()
            self.assertTrue(env.reset(), "The obs must be in observation space")
            
    def test_quantities_VRP(self):
        env = DynamicQVRPEnv(different_quantities=True, costs_KM=[1, 1], emissions_KM=[.1, .3])
        for _ in range(len(env.all_dests)):
            # state, _ = self.env.reset()
            self.assertTrue(env.reset(), "The obs must be in observation space")
            
    def test_vehicle_assignment(self):
        env = DynamicQVRPEnv(vehicle_assignment=True, costs_KM=[1, 1], emissions_KM=[.1, .3])
        for _ in range(len(env.all_dests)):
            # state, _ = self.env.reset()
            self.assertTrue(env.reset(), "The obs must be in observation space")
            while True:
                action = env.action_space.sample()
                state, _, done, _, info = env.step(action)
                self.assertTrue(True, 'The environment should not raise an error')
                self.assertTrue(info["remained_quota"] + 1e-4 >= 0, 'The quota must be respected')
                self.check_emissions_calculation(env.routes, info)
                self.check_routes_and_assignment(env.routes, env.assignment)
                
                if done:
                    break
                self.assertTrue(env.observation_space.contains(state), "The obs must be in observation space")
                
    def test_cluster_scenario(self):
        env = DynamicQVRPEnv(
            vehicle_assignment=True, costs_KM=[1, 1], emissions_KM=[.1, .3],
            cluster_scenario=True
            )
        for _ in range(len(env.all_dests)):
            # state, _ = self.env.reset()
            self.assertTrue(env.reset(), "The obs must be in observation space")
            while True:
                action = env.action_space.sample()
                state, _, done, trun, info = env.step(action)
                self.assertTrue(True, 'The environment should not raise an error')
                self.assertTrue(env.observation_space.contains(state), "The obs must be in observation space")
                self.assertTrue(info["remained_quota"] + 1e-4 >= 0, 'The quota must be respected')
                self.check_emissions_calculation(env.routes, info)
                self.check_routes_and_assignment(env.routes, env.assignment)
                if done or trun:
                    break
                
    def test_diffrent_DoDs(self):
        
        for DoD in [1., .75, .5, .25, 0]:
            env = DynamicQVRPEnv(
                DoD=DoD,
                costs_KM=[1, 1], emissions_KM=[.1, .3],
                )
            # state, _ = self.env.reset()
            s_idx = np.random.randint(len(env.all_dests))
            self.assertTrue(env.reset(s_idx), "The obs must be in observation space")
            while True:
                action = env.action_space.sample()
                state, _, done, trun, *_ = env.step(action)
                self.assertTrue(True, 'The environment should not raise an error')
                self.assertTrue(env.observation_space.contains(state), "The obs must be in observation space")
                if done or trun:
                    break

if __name__ == '__main__':
    unittest.main()