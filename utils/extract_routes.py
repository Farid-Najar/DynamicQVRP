from copy import deepcopy
from envs.assignment import AssignmentEnv, AssignmentGame, Package, RemoveActionEnv
import numpy as np
import pickle
import multiprocess as mp

def create_routes(env : AssignmentEnv, nb_routes = 5_000, retain_rate = 0., time_budget = 1, change_quantity = False):
    
    margin = 0
    nb_routes += margin
    def process(env: AssignmentEnv, id, q, retained_dests):
        np.random.seed(id)
        
        dests = retained_dests+list(np.random.choice(
            [
            i for i in range(env._game.grid_size**2)
                if i not in retained_dests
            ],
            size=env._game.num_packages - len(retained_dests)+1,
            replace=False
        ))
        assert len(dests) == len(set(dests))
        
        dests.remove(env._game.hub)
        assert len(dests) == env._game.num_packages
        
        quantities = np.ones(len(dests), dtype=int)
        if change_quantity:
            C = env._game.max_capacity * env._game.num_vehicles - env._game.num_packages
            c = (C*np.random.dirichlet(np.ones(len(dests)))).astype(int)
            quantities += c
        packages = [
            Package(
                destination=dests[k],
                quantity=quantities[k],
            )
            for k in range(len(dests))
        ]
        env.reset(packages = packages, time_budget = time_budget)
        res = {
            'd' : env.destinations,
            'r' : env.initial_routes
        }
        print(f'{id} done')
        q.put((id, res))
        return
        
    if retain_rate:
        comment = f'_retain{retain_rate}'
    else:
        comment = ''
    # if real_data:
    #     with open(f'./real_game_K{env._game.num_packages}{comment}.pkl', 'wb') as f:
    #         pickle.dump(env._game, f, -1)

    # else:
    with open(f'data/game_K{env._game.num_packages}{comment}.pkl', 'wb') as f:
        pickle.dump(env._game, f, -1)
    
    q = mp.Manager().Queue()
    
    env.reset()
    shape = env.initial_routes.shape
    routes = np.zeros((nb_routes, shape[0], shape[1]))
    destinations = np.zeros((nb_routes, env._game.num_packages))
    destinations[0] = env.destinations
    routes[0] = env.initial_routes
    retained_dests = list(np.random.choice(
        env.destinations,
        size=int(retain_rate*len(env.destinations)),
        replace=False
    ))
    retained_dests.append(env._game.hub)
    ps = []
    
    for i in range(1, nb_routes):
        ps.append(mp.Process(target = process, args = (deepcopy(env), i, q, retained_dests,)))
        ps[-1].start()
        
    for p in ps:
        p.join()
        
    print('all done !')
    while not q.empty():
        i, d = q.get()
        routes[i] = d['r']
        destinations[i] = d['d']
        
    
    delete = [l for l in range(len(destinations)) if len(destinations[l]) != len(set(destinations[l]))]
    
    destinations = np.delete(destinations, delete, 0)
    routes = np.delete(routes, delete, 0)

    destinations = destinations[:nb_routes-margin]
    routes = routes[:nb_routes-margin]
    print('operation succeeded!')
    
    # if real_data:
    #     np.save(f'real_routes_K{env._game.num_packages}{comment}', routes)
    #     np.save(f'real_destinations_K{env._game.num_packages}{comment}', destinations)

    # else:
    np.save(f'data/routes_K{env._game.num_packages}{comment}', routes)
    np.save(f'data/destinations_K{env._game.num_packages}{comment}', destinations)
    

def create_quantities(g, nb_routes = 5_000, time_budget = 1, real_data = False):
    
    margin = 0
    nb_routes += margin
    real = "real_" if real_data else ""
    def process(env: AssignmentEnv, id, q, dests):
        np.random.seed(id)
        
        quantities = np.ones(len(dests), dtype=int)
        C = env._game.max_capacity * env._game.num_vehicles - env._game.num_packages
        c = (C*np.random.dirichlet(np.ones(len(dests)))).astype(int)
        quantities += c
        packages = [
            Package(
                destination=dests[k],
                quantity=quantities[k],
            )
            for k in range(len(dests))
        ]
        env.reset(packages = packages, time_budget = time_budget)
        res = {
            'd' : dests,
            'q' : quantities,
            'r' : env.initial_routes
        }
        print(f'{id} done')
        q.put((id, res))
        return
    
    comment = f'_retain{1.}'
    with open(f'./{real}game_K{g.num_packages}{comment}.pkl', 'wb') as f:
        pickle.dump(g, f, -1)
    
    q = mp.Manager().Queue()
    env = AssignmentEnv(g)
    env.reset()
    shape = env.initial_routes.shape
    routes = np.zeros((nb_routes, shape[0], shape[1]))
    destinations = np.zeros((nb_routes, env._game.num_packages))
    quantities = np.ones((nb_routes, env._game.num_packages), dtype=int)
    destinations[0] = env.destinations
    routes[0] = env.initial_routes
    retained_dests = env.destinations
    
    for i in range(0, nb_routes, 5):
        ps = []
        env = AssignmentEnv(deepcopy(g))
        ps.append(mp.Process(target = process, args = (deepcopy(env), i, q, retained_dests.copy(),)))
        ps.append(mp.Process(target = process, args = (deepcopy(env), i+1, q, retained_dests.copy(),)))
        ps.append(mp.Process(target = process, args = (deepcopy(env), i+2, q, retained_dests.copy(),)))
        ps.append(mp.Process(target = process, args = (deepcopy(env), i+3, q, retained_dests.copy(),)))
        ps.append(mp.Process(target = process, args = (deepcopy(env), i+4, q, retained_dests.copy(),)))
        ps[-1].start()
        ps[-2].start()
        ps[-3].start()
        ps[-4].start()
        ps[-5].start()
        
        ps[-1].join()
        ps[-2].join()
        ps[-3].join()
        ps[-4].join()
        ps[-5].join()
        
        
    print('all done !')
    while not q.empty():
        i, d = q.get()
        routes[i] = d['r']
        destinations[i] = d['d']
        quantities[i] = d['q']
        
    for i in range(len(destinations)):
        if i >= len(destinations):
            break
        if len(destinations[i]) != len(set(destinations[i])):
            destinations = np.delete(destinations, i, 0)
            routes = np.delete(routes, i, 0)
            quantities = np.delete(quantities, i, 0)

    destinations = destinations[:nb_routes-margin]
    routes = routes[:nb_routes-margin]
    print('operation succeeded!')
    
    np.save(f'{real}routes_K{env._game.num_packages}{comment}', routes)
    np.save(f'{real}destinations_K{env._game.num_packages}{comment}', destinations)
    np.save(f'{real}quantities_K{env._game.num_packages}{comment}', quantities)
    

def test():
    g = AssignmentGame(
            grid_size=12,
            max_capacity=25,
            Q = 10,
            K=20,
            emissions_KM = [0., .1, .3, .3],
            costs_KM = [1, 1, 1, 1],
            seed=42
        )
    env = AssignmentEnv(g)
    np.random.seed(42)
    env.reset()
    dests = list(np.random.choice(
        env.destinations,
        size=20,
        replace=False
    ))

    assert len(dests) == len(set(dests))
    
    assert len(dests) == env._game.num_packages
    
    quantities = np.ones(len(dests), dtype=int)
    C = env._game.max_capacity * env._game.num_vehicles - env._game.num_packages
    c = (C*np.random.dirichlet(np.ones(len(dests)))).astype(int)
    quantities += c
    packages = [
        Package(
            destination=dests[k],
            quantity=quantities[k],
        )
        for k in range(len(dests))
    ]
    env.reset(packages = packages, time_budget = 5)
    print(env.quantities)
    

if __name__ == '__main__':
    n = 200
    
    # g = AssignmentGame(
    #         real_data=REAL,
    #         max_capacity=13,
    #         Q = 800,
    #         K=50,
    #         emissions_KM = [0., .1, .3, .3],
    #         costs_KM = [1, 1, 1, 1],
    #         seed=42
    #     )
    # env = AssignmentEnv(g)
    # create_routes(env, n, time_budget=30, real_data=REAL)
    
    g = AssignmentGame(
            max_capacity=25,
            Q = 100,
            K=25,
            emissions_KM = [.3],
            costs_KM = [1],
            seed=1917
        )
    env = AssignmentEnv(g)
    create_routes(env, n, time_budget=3)
    
    # g = AssignmentGame(
    #         max_capacity=13,
    #         Q = 800,
    #         K=50,
    #         emissions_KM = [0., .1, .3, .3],
    #         costs_KM = [1, 1, 1, 1],
    #         seed=42
    #     )
    # env = AssignmentEnv(g)
    # create_routes(env, n, time_budget=30, retain_rate=.8)
    
    
    
    # g = AssignmentGame(
    #         max_capacity=25,
    #         Q = 900,
    #         K=100,
    #         emissions_KM = [0., .1, .3, .3],
    #         costs_KM = [1, 1, 1, 1],
    #         seed=42
    #     )
    # env = AssignmentEnv(g)
    # create_routes(env, n, time_budget=60)
    
    # g = AssignmentGame(
    #         max_capacity=25,
    #         Q = 500,
    #         K=20,
    #         emissions_KM = [0., .1, .3, .3],
    #         costs_KM = [1, 1, 1, 1],
    #         seed=42
    #     )
    # create_quantities(g, n, time_budget=30)