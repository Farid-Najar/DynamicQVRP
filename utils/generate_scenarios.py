import numpy as np


def create_random_scenarios(
    n_scenarios = 500,
    d = 50,
    hub = 0
):
    dests = np.zeros((n_scenarios, d), np.int64)
    p = np.load("data/prob_dests.npy")
    p[hub] = 0
    p /= p.sum()
    
    for i in range(n_scenarios):
        dests[i] = np.random.choice(
            range(len(p)),
            size = d,
            replace=False,
            p = p
        )
    
    np.save(f'data/destinations_K{d}_{n_scenarios}', dests)
    
if __name__ == '__main__':
    create_random_scenarios()