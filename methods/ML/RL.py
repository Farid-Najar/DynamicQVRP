import numpy as np
# from tqdm import tqdm
# import multiprocess as mp
from copy import deepcopy

# from gymnasium import Env
# from methods import Agent

# import gymnasium as gym
import math
import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt
import matplotlib

from stable_baselines3 import PPO, DQN
# from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from methods.ML.supervised import NN

from time import time


import sys
import logging
import os
from pathlib import Path
path = Path(os.path.dirname(__file__))
sys.path.insert(1, str(path.parent.absolute()))

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def mask_geography(observation, n_vehicles):
    observation[n_vehicles+2:n_vehicles+2+(n_vehicles+1)] = 0.
    return observation

def mask_horizon(observation, n_vehicles):
    observation[n_vehicles+1] = 0.
    return observation

def noisy_horizon(observation, n_vehicles, noise = 0.1):
    observation[n_vehicles+1] += np.random.normal(0, noise)
    observation[n_vehicles+1] = np.clip(observation[n_vehicles+1], 0, 1)
    return observation

def plot_durations(rs, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(rs, dtype=torch.float)
    ones_t = torch.ones_like(durations_t, dtype=torch.float)
    # offline_t = torch.tensor(offline, dtype=torch.float)
    # greedy_t = torch.tensor(greedy, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Mean Rewards')
    plt.plot(durations_t.numpy(), label = 'DQN')
    plt.plot(ones_t.numpy())
    # plt.plot(offline_t.numpy(), label = 'Offline')
    # plt.plot(greedy_t.numpy(), label = 'Greedy')
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(10)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, 
                #  observation_space: spaces.Box, 
                 n_observation, 
                 hidden_layers = [512, 512, 256],#[1024, 1024, 512, 256] [512, 512, 256]
                 n_actions: int = 2):
        # super().__init__(observation_space, n_actions)
        super().__init__()
        
        hidden_layers.insert(0, n_observation)
        layers = []
        for l in range(len(hidden_layers)-1):
            layers += [
                nn.Linear(hidden_layers[l], hidden_layers[l+1]),
                nn.ReLU()
            ]
            
        layers += [
            nn.Linear(hidden_layers[-1], n_actions),
            # nn.Sigmoid()
            # nn.Softmax()
        ]

        self.linear = nn.Sequential(
            *layers
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(observations)
    
    
class LinearDQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer = nn.Linear(n_observations, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        return self.layer(x)

# class LinearQRL(Agent):
    
#     def __init__(self, env: Env, n_workers=5, parallelize=False, *args, **kwargs):
#         super().__init__(env, n_workers, parallelize, *args, **kwargs)
        
#         self.device = torch.device(
#             "cuda" if torch.cuda.is_available() else
#             "mps" if torch.backends.mps.is_available() else
#             "cpu"
#         )
        
def train_DQN(
    env,
    test_env = None,
    hidden_layers = [1024, 1024, 1024],
    EPISODES = 20,
    BATCH_SIZE = 128,
    GAMMA = 0.99,
    EPS_START = 1.,
    EPS_END = 0.05,
    EPS_DECAY = 1000, # in episodes
    TAU = 0.01,
    update_target_every = 100,
    LR = 1e-4,
    model_path = 'model_DQN',
    eval_every = 200,
    save = True,
    mask_geography_flag = False,
    mask_horizon_flag = False,
):
    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer
    
    # if GPU is to be used
    device = torch.device(
        # "cuda" if torch.cuda.is_available() else
        # "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    
    EPS_DECAY = EPS_DECAY*env.H # for adaptive : *(env.action_space.n-1)#num_episodes*env.H//2
    
    if test_env is None:
        test_env = deepcopy(env)
    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(state)

    policy_net = NN(n_observations, deepcopy(hidden_layers), n_actions).to(device)
    target_net = NN(n_observations, deepcopy(hidden_layers), n_actions).to(device)
    # print(policy_net)
    # print(target_net)
    # assert False
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    global steps_done
    steps_done = 0
    
    def select_action(state, deterministic = False):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold or deterministic:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        
    
    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        # criterion = nn.SmoothL1Loss()

        # Compute L2 loss
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()
        
    
    def eval():
        
        rs = np.zeros(len(test_env.all_dests))
        for i_episode in range(len(test_env.all_dests)):
            # Initialize the environment and get its state
            state, info = test_env.reset(i_episode)
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            episode_r = 0.
            
            while True:
                action = select_action(state, deterministic=True)
                state, reward, terminated, truncated, _ = test_env.step(action.item())
                # reward = np.array(reward, np.float32)
                # print(reward[None])
                # reward = torch.tensor(reward[None], device=device)
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                
                done = terminated or truncated
                episode_r += reward
                if done:
                    break
                
            rs[i_episode] = episode_r
            
        return rs

    best_r = 0.
    n_scenarios = len(env.all_dests)
    num_episodes = EPISODES#n_scenarios*EPOCHS
    epoch_r = np.zeros(n_scenarios)
    
    test_rs = []
    test_r = eval()
    test_rs.append(test_r)
    mean_r = test_r.mean()
    print(f"0 : {mean_r:.3f}")
    
    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state, _ = env.reset()
        if mask_geography_flag:
            state = mask_geography(state, env.E.shape[0])
        if mask_horizon_flag:
            state = mask_horizon(state, env.E.shape[0])
            
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_r = 0.
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            if mask_geography_flag:
                observation = mask_geography(observation, env.E.shape[0])
            if mask_horizon_flag:
                observation = mask_horizon(observation, env.E.shape[0])
                
            reward = np.array(reward, np.float32)
            # print(reward[None])
            reward = torch.tensor(reward[None], device=device)
            done = terminated or truncated
            episode_r += reward

            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()
            
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
                
            if i_episode%update_target_every == 0:
                # Periodic hard update of the target network's weights
                
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]
                target_net.load_state_dict(target_net_state_dict)
            else:
                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_net.load_state_dict(target_net_state_dict)

            if done:
                epoch_r[i_episode%n_scenarios] = episode_r#/(res_greedy["rs"][i_episode%200]+1e-8)
                
                # offline.append(res_offline["rs"][i_episode%200])
                # greedy.append(res_greedy["rs"][i_episode%200])
                # print(i_episode, episode_r)
                if (i_episode+1)%eval_every == 0:
                    test_r = eval()
                    test_rs.append(test_r)
                    mean_r = test_r.mean()
                    print(i_episode+1, f" : {mean_r:.3f}")
                    
                    if best_r < mean_r:
                        best_r = mean_r
                        print(f'best ! mean rewards : {best_r:.3f}')
                        if save :
                            # model_path = 'model_{}_{}'.format(timestamp, epoch)
                            # model_path = 'model_DQN'
                            torch.save(policy_net.state_dict(), f'model_{model_path}')
                # if (i_episode+1)%n_scenarios == 0:
                #     episode_durations.append(epoch_r.copy())
                    # Track best performance, and save the model's state
                    

                    # plot_durations()
                break

    print('Complete')       
    np.save(f'results/rewards_{model_path}', np.array(test_rs))
    return test_rs


class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PPO, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist, value

class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.policy = PPO(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr, amsgrad=True)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.policy_old = PPO(state_dim, action_dim, hidden_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        dist, _ = self.policy_old(state)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)
        old_states = torch.tensor(memory.states, dtype=torch.float32)
        old_actions = torch.tensor(memory.actions, dtype=torch.float32)
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32)

        for _ in range(self.K_epochs):
            dist, state_values = self.policy(old_states)
            state_values = state_values.squeeze()
            dist_entropy = dist.entropy()
            new_logprobs = dist.log_prob(old_actions)
            ratios = torch.exp(new_logprobs - old_logprobs)

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
    
    def train(self, env, num_episodes, max_timesteps):
        
        for episode in range(num_episodes):
            memory = Memory()
            state = env.reset()
            for t in range(max_timesteps):
                action, log_prob = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                
                memory.states.append(state)
                memory.actions.append(action)
                memory.logprobs.append(log_prob)
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                
                state = next_state
                
                if done:
                    break
            
            self.update(memory)
            memory.clear_memory()
            print(f"Episode {episode+1}/{num_episodes} completed")
    
    # Example usage:
    # agent = PPOAgent(state_dim, action_dim)
    # agent.train(env, num_episodes=1000, max_timesteps=200)
        
def make_env(envs, rank: int, seed: int = 1917):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env2 = deepcopy(envs[rank%len(envs)])
        env2.reset(rank, seed=seed + rank)
        return env2
    set_random_seed(seed)
    return _init

def _train(
    vec_env,
    algo = PPO,
    algo_kwargs = dict(
        policy = "MlpPolicy",
        policy_kwargs = {},
        gamma = 0.99,
    ),
    # algo_file : str|None = None,
    algo_dir = str(path)+'/ppo',
    budget = 1000,
    n_eval = 10,
    save = True,
    eval_freq = 200,
    progress_bar =True,
):
    
    # Instantiate the agent
    # if algo_file is not None:
    #     try:
    #         model = algo.load(algo_file+f'/{algo.__name__}', env=vec_env)
    #         assert model.policy_kwargs == policy_kwargs
    #     except Exception as e:
    #         logging.warning(f'couldnt load the model because this exception has been raised :\n{e}')
            
    #         print(f'path is {path}')
    #         raise('couldnt load the model!')
    # else:   
    model = algo(**algo_kwargs)

    logging.info(f"the model parameters :\n {model.__dict__}")
    # Train the agent and display a progress bar
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=n_eval)
    logging.info(f'Before training :\n mean, std = {mean_reward}, {std_reward}')
    
    # checkpoint_callback = CheckpointCallback(
    # #   save_freq=1000,
    #   save_path="./logs/",
    #   name_prefix="rl_model",
    #   save_replay_buffer=True,
    #   save_vecnormalize=True,
    # )
    
    eval_callback = EvalCallback(vec_env, best_model_save_path=algo_dir,
                             log_path=algo_dir, eval_freq=eval_freq,
                             deterministic=True)
    
    model.learn(
        total_timesteps=budget,
        progress_bar=progress_bar,
        log_interval=1000,
        callback=eval_callback,
        # tb_log_name="ppo",
    )
    # Save the agent
    if save:
        model.save(f'{str(path)}/models/{algo.__name__}')
    # del model  # delete trained model to demonstrate loading
    return model


def train_RL(
    envs,
    algo = PPO,
    net_arch : list =[512, 512, 256],
    budget = int(2e4),
    save = True,
    save_path = None,
    eval_freq = 200,
    progress_bar =True,
    algo_file = None,
    n_steps = 32,
    gamma = 0.99,
    *args,
    **kwargs
):
    
    if save_path is None:
        save_path = str(path)+f'/models/{algo.__name__}'
        
    logging.basicConfig(
        filename=save_path+f'/{algo.__name__}.log',
        filemode='w',
        format='%(levelname)s - %(name)s :\n%(message)s \n',
        level=logging.INFO,
    )
    logging.info(f'Train {algo.__name__} started !')
    # Create environment
    n_eval = len(envs[0].all_dests)
    # check_env(env)
    # logging.info(
    #     f"""
    #     Environment information :
        
    #     Q = {env.Q}
    #     K = {env.K}
    #     total capacity = {env.total_capacity} 
    #     """
    # )
    # Create environment
    num_cpu = os.cpu_count()
    logging.info(f'Number of CPUs = {num_cpu}')
    vec_env = SubprocVecEnv([make_env(envs, i) for i in range(num_cpu-1)])
    vec_env = VecMonitor(vec_env, save_path+"/")
    # log(type(env))
    
    if algo == PPO:
        algo_kwargs = dict(
            env = vec_env,
            policy_kwargs = dict(
                activation_fn=nn.ReLU,
                share_features_extractor=True,
                net_arch=net_arch.copy(),
                optimizer_class = optim.AdamW,
                optimizer_kwargs = dict(
                    amsgrad=True
                ),
            ),
            n_steps = 128,
            batch_size=n_steps*(os.cpu_count()-1),
            # learning_rate=5e-5, #TODO change
            verbose=1,
            gamma=gamma,
            tensorboard_log=save_path+"/"
            
        )
    else:
        algo_kwargs = dict(
            env = vec_env,
            policy = "MlpPolicy",
            policy_kwargs = dict(
                activation_fn=nn.ReLU,
                net_arch=net_arch.copy(),
                optimizer_class = optim.AdamW,
                optimizer_kwargs = dict(
                    amsgrad=True
                ),
            ),
            target_update_interval=5000,
            tau=.05,
            batch_size=128,
            buffer_size=10_000,
            learning_rate=1e-4,
            exploration_fraction=0.1,
            verbose=1,
            gamma=gamma,
            
        )
    
    model = _train(
        vec_env,
        algo_kwargs=algo_kwargs,
        algo=algo,
        budget=budget,
        n_eval=n_eval,
        save = save,
        algo_dir=save_path,
        eval_freq =     eval_freq ,
        progress_bar =    progress_bar,
        # algo_file = algo_file,
    )

    

    

    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `log_system_info=True`
    # to compare the system on which the model was trained vs the current one
    # model = DQN.load("dqn_lunar", env=env, log_system_info=True)
    # model = PPO.load("ppo", env=env)

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    logging.info(f'After training :\n mean, std = {mean_reward}, {std_reward}')
    
    return model
    