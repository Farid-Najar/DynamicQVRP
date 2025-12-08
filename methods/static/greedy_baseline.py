import numpy as np

from envs import StaticQVRPEnv
import pickle
from time import time
import multiprocess as mp
from copy import deepcopy

def baseline(
    env : StaticQVRPEnv,
    *args,
    **kwargs,
    ):
    # vehicles_by_priority = np.argsort(game.emissions_KM)
    
    # q = np.ones(len(game.packages)+1)
    # q[1:] = np.array([
    #     d.quantity for d in game.packages
    # ])
    
    # nodes = [
    #     d.destination for d in game.packages
    # ]
    
    num_packages = env._env.H
    
    action = np.ones(num_packages, dtype=int)
    rewards = np.zeros(num_packages)
    excess_emission = np.zeros(num_packages)
    infos = []#[dict() for _ in range(len(action))]
    # omitted = np.zeros(game.num_packages)
    
    best = action.copy()
    solution = best.copy()
    
    # l = [game.hub] + nodes
    # x, y = np.ix_(l, l)
    
    # infos = []
    
    # A = game.distance_matrix[x, y]@np.diag(1/q)
    # indices = np.flip(np.argsort(np.mean(A[1:, 1:] + np.max(A[1:, 1:])*np.eye(len(A[1:, 1:])), axis=1)))
    _, r_best, d, _, info = env.step(action, simulation = True)
    emission = info['excess_emission']
    # r_best = float('-inf') if d else r_best
    # o = info['omitted']
    
    indices = list(range(num_packages))
    
    for t in range(num_packages):
        excess_emission[t] = emission
        # omitted[t] = o
        rewards[t] = r_best
        
        if d:
            print('done : ', t, emission)
            break
        
        r_best = float('-inf')
        
        for i in indices:
            a = action.copy()
            a[i] = 1 - a[i]
            _, r,_, _, info = env.step(a, simulation = True)#, time_budget, call_OR=(full_OR and t%OR_every == 0))
            # action = np.ones(game.num_packages, dtype=bool)
            
            if r > r_best:
                emission = info['excess_emission']
                # o = info['omitted']
                r_best = r
                if r > np.max(rewards[:t+1]):
                    solution = a.copy()
                best = a.copy()
                infos.append(info)
                ii = i
                
        # print(len(indices))
        indices.remove(ii)
        action = best.copy()
        _, r_best, d, _, info = env.step(action)
        r_best = float('-inf') if d else r_best
        # if d:
        #     break
        # r_best = float('-inf') if d else r_best

        
        # infos.append(info)
    # for m in vehicles_by_priority:
    #     frm = game.hub
    
    res = {
        'solution' : solution,
        'rewards' : rewards,
        'excess_emission' : excess_emission,
        'infos' : infos,
    }
    
    return res



def baseline2(
    env : StaticQVRPEnv,
    *args,
    **kwargs,
    ):
    # vehicles_by_priority = np.argsort(game.emissions_KM)
    
    # q = np.ones(len(game.packages)+1)
    # q[1:] = np.array([
    #     d.quantity for d in game.packages
    # ])
    
    # nodes = [
    #     d.destination for d in game.packages
    # ]
    
    num_packages = env._env.H
    
    action = np.ones(num_packages, dtype=bool)
    rewards = np.zeros(num_packages)
    excess_emission = np.zeros(num_packages)
    infos = []#[dict() for _ in range(len(action))]
    # omitted = np.zeros(game.num_packages)
    
    best = action.copy()
    solution = best.copy()
    
    # l = [game.hub] + nodes
    # x, y = np.ix_(l, l)
    
    # infos = []
    
    # A = game.distance_matrix[x, y]@np.diag(1/q)
    # indices = np.flip(np.argsort(np.mean(A[1:, 1:] + np.max(A[1:, 1:])*np.eye(len(A[1:, 1:])), axis=1)))
    obs, r_best, d, _, info = env.step(action.astype(int), simulation = True)
    emission = info['excess_emission']
    # o = info['omitted']
    
    indices = list(range(num_packages))
    
    for t in range(num_packages):
        excess_emission[t] = emission
        # omitted[t] = o
        rewards[t] = r_best
        
        if d:
            break
        
        r_best = float('-inf')
        
        # for i in indices:
        #     a = action.copy()
        #     a[i] = not a[i]
        #     _, r,_, _, info = env.step(a.astype(int), simulation = True)#, time_budget, call_OR=(full_OR and t%OR_every == 0))
        #     # action = np.ones(game.num_packages, dtype=bool)
            
        #     if r > r_best:
        #         emission = info['excess_emission']
        #         # o = info['omitted']
        #         r_best = r
        #         if r > np.max(rewards[:t+1]):
        #             solution = a.copy()
        #         best = a.copy()
        #         infos.append(info)
        #         ii = i
                
        # print(len(indices))
        obs -= env._env.omission_cost*env._env.quantities
        a = np.argmax(obs)
        indices.remove(a)
        best[a] = 0
        action = best.copy()
        _, r_best, d, _, info = env.step(action.astype(int))
        emission = info['excess_emission']
# def simulate(
#     n_simulation = 100,
#     Q = 30,
#     K = 50,
#     T = 500,
#     full_OR = False,
#     OR_every = 1,
#     ):
#     def process_baseline(game, id, q):
#         t0 = time()
#         res = baseline(game, full_OR=full_OR, OR_every=OR_every)
#         res['time'] = time() - t0
#         q.put((id, res))
#         print(f'baseline {id} done')
        
#     q = mp.Manager().Queue()
#     res = dict()
    
#     ps = []

#     for i in range(n_simulation):
#         game = AssignmentGame(Q=30, K = 50)
#         game.reset()
#         # threads.append(Thread(target = process, args = (game, res[i])))
#         ps.append(mp.Process(target = process_baseline, args = (deepcopy(game), i, q,)))
#         ps[i].start()
        
#     for i in range(n_simulation):
#         ps[i].join()
        
#     print('all done !')
#     while not q.empty():
#         i, d = q.get()
#         res[i] = d
        
#     with open(f"res_fullOR{OR_every}_baseline_Q{Q}_K{K}_n{n_simulation}_T{T}.pkl","wb") as f:
#         pickle.dump(res, f)
        
# if __name__ == '__main__':
    
    # simulate(n_simulation = 100, full_OR=True, OR_every = 10)
    # simulate(n_simulation = 100, full_OR=True, OR_every = 5)
    # simulate(n_simulation = 100, full_OR=True, OR_every = 1)
    
    
    # q = (np.random.randint(5, size=(5)) + 1)
    # Q = np.diag(1/q)
    # A = np.random.randint(50, size=(5, 5))
    # np.fill_diagonal(A,0)
    # B = A@np.diag(1/q)

    # print(np.all(B == A@Q))
    # import matplotlib.pyplot as plt
    
    # game = AssignmentGame()#Q = 50, max_capacity=150, grid_size=25)
    # game.reset(50)
    
    # res = greedy(game, time_budget=2)
    # # print(best)
    # # rg, eg = greedy(game, time_budget=1)
    
    # fig, ax1 = plt.subplots()

    # color = 'tab:red'
    # ax1.set_ylabel('rewards', color=color)
    # ax1.plot(res['rewards'], color=color)
    # ax1.tick_params(axis='y', labelcolor=color)

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # color = 'tab:blue'
    # ax2.set_ylabel('excess emissions', color=color)  # we already handled the x-label with ax1
    # ax2.plot(res['excess_emission'], color=color)
    # ax2.tick_params(axis='y', labelcolor=color)

    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.title('Results median')
    # plt.show()
    
    # plt.plot(om)
    # plt.show()
    
    # fig, ax1 = plt.subplots()
    
    # color = 'tab:red'
    # ax1.set_ylabel('rewards', color=color)
    # ax1.plot(rg, color=color)
    # ax1.tick_params(axis='y', labelcolor=color)

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # color = 'tab:blue'
    # ax2.set_ylabel('excess emissions', color=color)  # we already handled the x-label with ax1
    # ax2.plot(eg, color=color)
    # ax2.tick_params(axis='y', labelcolor=color)

    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.title('Results greedy')
    # plt.show()