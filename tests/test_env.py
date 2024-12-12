import unittest

from envs import DynamicQVRPEnv
from stable_baselines3.common.env_checker import check_env

class TestEnv(unittest.TestCase):

    def test_sb3(self):
        env = DynamicQVRPEnv()
        check_env(env)
        
if __name__ == '__main__':
    unittest.main()