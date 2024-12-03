import numpy as np
import os
print(os.getcwd())
from envs import DynamicQVRPEnv
import pickle
from tqdm import tqdm

def generate_xy(file = "res_wReOpt_500"):
    
    with open(f'results/{file}/res_offline.pkl', "rb") as f:
        res_offline = pickle.load(f)
        
    with open(f'results/{file}/env_configs.pkl', "rb") as f:
        env_configs = pickle.load(f)
    x = []
    y = []
    
    env = DynamicQVRPEnv(**env_configs)
    
    for i in tqdm(range(len(res_offline["actions"]))):
        actions = res_offline["actions"][i]
        o, _ = env.reset(i)
        x.append(o)
        for a in actions:
            y.append(int(bool(a)))
            o, *_, d, _ = env.step(int(a))
            if d:
                break
            x.append(o)
            
        # x.pop(-1)
    assert len(x) == len(y)
    x = np.array(x)
    y = np.array(y)
    
    np.save('data/SL_data/x.npy', x)
    np.save('data/SL_data/y.npy', y)
    

if __name__ == '__main__':
    generate_xy()