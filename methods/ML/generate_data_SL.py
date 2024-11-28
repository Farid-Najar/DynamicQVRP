import numpy as np
import os
print(os.getcwd())
from envs import DynamicQVRPEnv
import pickle

def generate_xy():
    with open('results/res_w_ReOpt/res_offline.pkl', "rb") as f:
        res_offline = pickle.load(f)
    x = []
    y = []
    
    env = DynamicQVRPEnv(50, 100, DoD=0.5, vehicle_capacity=25, re_optimization=True,
                          costs_KM=[1, 1], emissions_KM=[.1, .3]
    )
    for i in range(len(res_offline["actions"])):
        actions = res_offline["actions"][i]
        o, _ = env.reset(i)
        x.append(o)
        for a in actions:
            y.append(a)
            o, *_, d, _ = env.step(int(a))
            if d:
                break
            x.append(o)
            
        # x.pop(-1)
    assert len(x) == len(y)
    x = np.array(x)
    y = np.array(y)
    
    np.save('data/SL_data/x_50.npy', x)
    np.save('data/SL_data/y_50.npy', y)
    

if __name__ == '__main__':
    generate_xy()