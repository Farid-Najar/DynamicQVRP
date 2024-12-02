import sys
import os
from copy import deepcopy
# direc = os.path.dirname(__file__)
from pathlib import Path
path = Path(os.path.dirname(__file__))
# pri&
# caution: path[0] is reserved for script path (or '' in REPL)
# print(str(path)+'/ppo')
sys.path.insert(1, str(path.parent.absolute()))

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from gymnasium import spaces
import gymnasium as gym
import numpy as np

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class NN(nn.Module):#(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, 
                #  observation_space: spaces.Box, 
                 n_observation, 
                 hidden_layers = [512, 512, 256],#[1024, 1024, 512, 256] [512, 512, 256]
                 n_actions: int = 1):
        # super().__init__(observation_space, n_actions)
        super().__init__()
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # n_inputs = observation_space.shape[0]
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
    

def train(x: torch.Tensor, y: torch.Tensor, epochs = 10, save = False):
    # mps_device = torch.device("mps")
    
    device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
    )
    
    pivot = int(.8*len(x))
    y = y.view(-1, 1)
    print(y.shape)
    print(x[:pivot, :].shape)
    print(y[:pivot, :].shape)
    training_set = TensorDataset(x[:pivot, :], y[:pivot, :])
    validation_set = TensorDataset(x[pivot:, :], y[pivot:, :])
    
    model = NN(x.shape[1]).to(device)
    loss_fn = nn.MSELoss()#nn.CrossEntropyLoss()#nn.BCELoss(reduction='mean')#
    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = DataLoader(
        training_set, batch_size=256, shuffle=True, num_workers=os.cpu_count()-1,
    )
    validation_loader = DataLoader(
        validation_set, batch_size=256, shuffle=False, num_workers=os.cpu_count()-1,
    )

    optimizer = torch.optim.Adam(model.parameters(),lr=5e-4,  weight_decay=1e-5)
    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            # inputs.to(mps_device)
            # labels.to(mps_device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs.to(device)).to(device)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels.to(device))
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 8 == 7:
                last_loss = running_loss / 8 # loss per batch
                # print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

    best_vloss = 1_000_000.

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch, writer)


        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                # vinputs.to(mps_device)
                # vlabels.to(mps_device)
                voutputs = model(vinputs.to(device))#.round().to(mps_device)
                vloss = loss_fn(voutputs, vlabels.to(device))#torch.sum(voutputs == vlabels.to(mps_device))/vlabels.nelement()
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            print('best ! loss :', best_vloss)
            if save :
                # model_path = 'model_{}_{}'.format(timestamp, epoch)
                model_path = 'model_SL'
                torch.save(model.state_dict(), model_path)

    

if __name__ == '__main__':
    # mps_device = torch.device("mps")
    # torch.set_default_device(mps_device)
    
    # path = 'DynamicQVRP/data/SL_data/'
    path = 'data/SL_data/'
    
    x = np.load(path+'x_downsampled_50.npy')
    y = np.load(path+'y_downsampled_50.npy')
    # x /= np.amax(x, axis=1).reshape(-1, 1)
    assert len(x) == len(y)
    print(x.shape)
    
    train(torch.Tensor(x), torch.Tensor(y), 500, save=True)