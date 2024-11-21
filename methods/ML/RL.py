import numpy as np
from tqdm import tqdm
import multiprocess as mp
from copy import deepcopy

from gymnasium import Env
from methods import Agent

import torch

class LinearQRL(Agent):
    
    def __init__(self, env: Env, n_workers=5, parallelize=False, *args, **kwargs):
        super().__init__(env, n_workers, parallelize, *args, **kwargs)
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        
        