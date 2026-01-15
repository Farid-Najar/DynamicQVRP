import numpy as np


def create_random_scenarios(
    n_scenarios = 500,
    d = 50,
    hub = 0,
    save = True,
    p = None,
    comment = ''
    ):
    dests = np.zeros((n_scenarios, d), np.int64)
    p = p if p is not None else np.load("data/prob_dests.npy")
    p[hub] = 0
    p /= p.sum()
    
    for i in range(n_scenarios):
        dests[i] = np.random.choice(
            range(len(p)),
            size = d,
            replace=False,
            p = p
        )
    if save:
        np.save(f'data/destinations_K{d}_{n_scenarios}{comment}', dests)
    return dests

def create_noised_random_scenarios(
    n_scenarios = 500,
    d = 50,
    noise_level = 0.025,
    hub = 0,
    save = True,
    p = None,
    ):
    dests = np.zeros((n_scenarios, d), np.int64)
    p = p if p is not None else np.load("data/prob_dests.npy")
    p += np.random.normal(0, noise_level, len(p))
    exp_p = np.exp(p)
    p = exp_p/exp_p.sum()
    p[hub] = 0
    p /= p.sum()
    
    for i in range(n_scenarios):
        dests[i] = np.random.choice(
            range(len(p)),
            size = d,
            replace=False,
            p = p
        )
    if save:
        np.save(f'data/noised_destinations_K{d}_{n_scenarios}', dests)
    return dests
    
if __name__ == '__main__':
    
    create_random_scenarios(
        d=20, n_scenarios=100, p = np.ones(82),
        comment='_essonne_test'
    )
    
    create_random_scenarios(
        d=20, n_scenarios=500, p = np.ones(82),
        comment='_essonne'
    )
    # create_random_scenarios(
    #     d=50, n_scenarios=100, p = np.ones(500),
    #     comment='_uniforme_test'
    # )
    # create_noised_random_scenarios(d=50, n_scenarios=100)
    
    # Uniform data set
    # create_random_scenarios(
    #     d=100, n_scenarios=100, p = np.ones(500),
    #     comment='_uniform_test'
    # )
    
    # create_random_scenarios(
    #     d=100, n_scenarios=500, p = np.ones(500),
    #     comment='_uniform'
    # )