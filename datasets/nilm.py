import numpy as np
import torch

import torchvision
from torchvision import transforms

from pathlib import Path

import ai8x

PROFILE_NAME = "unetnilm_ukdale_20240321_155419"
EXTENSION_MODE = "randomizer"

def split_ukdale(data):
    if EXTENSION_MODE == "appender":
        split_1 = int(0.15 * len(data))
        split_2 = int(0.75 * len(data))
        validation = data[:split_1]
        train = data[split_1:split_2]
        test = data[split_2:]
    else:
        split_1 = int(0.60 * len(data))
        split_2 = int(0.85 * len(data))
        train = data[:split_1]
        validation = data[split_1:split_2]
        test = data[split_2:]
    return train, validation, test

def load_data(data_path, data_type="training", sample=None, data="ukdale", denoise=False):
    if denoise:
        x = np.load(data_path+f"/{data}/{PROFILE_NAME}/{data_type}/denoise_inputs.npy")
    else:
        x = np.load(data_path+f"/{data}/{PROFILE_NAME}/{data_type}/noise_inputs.npy")
    y = np.load(data_path+f"/{data}/{PROFILE_NAME}/{data_type}/targets.npy")
    z = np.load(data_path+f"/{data}/{PROFILE_NAME}/{data_type}/states.npy")
    if sample is None:
        return x, y, z
    else:
        return x[:sample], y[:sample], z[:sample]

class Dataset(torch.utils.data.Dataset):
    def __init__(self,  inputs, targets, states,  seq_len=99):
        self.inputs = inputs
        self.targets = targets
        self.states  = states
        seq_len = seq_len  if seq_len% 2==0 else seq_len+1
        self.seq_len = seq_len
        self.len = self.inputs.shape[0] - self.seq_len
        self.indices = np.arange(self.inputs.shape[0])

    def __len__(self):
        'Denotes the total number of samples'
        return self.len
    
    def get_sample(self, index):
        indices = self.indices[index : index + self.seq_len]
        inds_inputs=sorted(indices[:self.seq_len])
        inds_targs=sorted(indices[self.seq_len-1:self.seq_len])

        return self.inputs[inds_inputs], self.targets[inds_targs], self.states[inds_targs]

    def __getitem__(self, index):
        inputs, target, state = self.get_sample(index)
        return torch.tensor(inputs).unsqueeze(-1).float(), torch.tensor(target).float().squeeze(), torch.tensor(state).long().squeeze()

class UkdaleDataset(Dataset):

    def __init__(self, root_dir, d_type, transform=None, seq_len=99, denoise=False, data_type="training"):
        
        inputs, targets, states = load_data(data_path=str(root_dir),
                                data="ukdale",
                                data_type=data_type,
                                denoise=denoise
                                )
        inputs_train, inputs_val, inputs_test = split_ukdale(inputs)
        targets_train, targets_val, targets_test = split_ukdale(targets)
        states_train, states_val, states_test = split_ukdale(states)
        
        if d_type == "train":
            inputs = inputs_train
            targets = targets_train
            states = states_train
        elif d_type == "test":
            inputs = inputs_test
            targets = targets_test
            states = states_test
        elif d_type == "val":
            inputs = inputs_val
            targets = targets_val
            states = states_val

        super().__init__(inputs, targets, states, seq_len)

        self.transform = transform

    def visualize_batch(self, index=None):

        index = 0 if index is None else index

        print(index)

def ukdale_get_datasets(data, load_train=True, load_test=True, load_val=None):

    (data_dir, args) = data

    if load_train:
        train_transform = transforms.Compose([])
        
        train_dataset = UkdaleDataset(root_dir=data_dir,
                                      d_type="train",
                                      transform=train_transform)
        
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([])

        test_dataset = UkdaleDataset(root_dir=data_dir,
                                     d_type="test",
                                     transform=test_transform)
    else:
        test_dataset = None
    
    if load_val:
        val_transform = transforms.Compose([])

        val_dataset = UkdaleDataset(root_dir=data_dir,
                                     d_type="val",
                                     transform=val_transform)
    else:
        val_dataset = None

    if load_val:
        return train_dataset, test_dataset, val_dataset
    else:
        return train_dataset, test_dataset

datasets = [
    {
        'name': 'ukdale',
        'input' : (128),
        'output' : (),
        'loader' : ukdale_get_datasets,
    }
]