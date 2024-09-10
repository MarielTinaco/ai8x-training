import numpy as np
import torch

import torchvision
from torchvision import transforms
from sklearn.preprocessing import minmax_scale

from pathlib import Path

import ai8x

class Seq2PointNILM(torch.utils.data.Dataset):

    def __init__(self, root, d_type, seq_len=100, transform=None,denoise=True):
        self.root = Path(root)
        self.d_type = d_type
        self.transform = transform
        self.seq_len = seq_len

        self.data, self.labels = self.__load(denoise=denoise)
        self.indices = np.arange(self.data.shape[0])

    def __load(self, denoise=True):

        if denoise:
            x = np.load(self.root / "NILMTK" / "processed" / self.d_type / "denoise_inputs.npy" )
        else:
            x = np.load(self.root / "NILMTK" / "processed" / self.d_type / "noise_inputs.npy" )
        y = np.load(self.root / "NILMTK" / "processed" / self.d_type / "targets.npy" )
        z = np.load(self.root / "NILMTK" / "processed" / self.d_type / "states.npy" )

        x = minmax_scale(x, feature_range=(0, 1))
        z = np.apply_along_axis(lambda data : minmax_scale(data, feature_range=(0, 1)), 0, z)

        return x, (y, z)

    def __reshape_data(self, audio, row_len=100):
        # add overlap if necessary later on
        return torch.transpose(audio.reshape((-1, row_len)), 1, 0)

    def __len__(self):
        return self.data.shape[0] - self.seq_len

    def get_sample(self, index):
        indices = self.indices[index : index + self.seq_len]
        inds_inputs=sorted(indices[:self.seq_len])
        inds_targs=sorted(indices[self.seq_len-1:self.seq_len])
        
        states = self.labels[0]
        power = self.labels[1]

        return self.data[inds_inputs], (states[inds_targs], power[inds_targs])

    def __getitem__(self, index):
        inputs, targets = self.get_sample(index)
        state = targets[0]
        power = targets[1]

        inputs = torch.tensor(inputs)
        inputs = self.__reshape_data(inputs, row_len=self.seq_len)
        inputs = inputs.type(torch.FloatTensor)

        if self.transform:
            inputs = self.transform(inputs)

        power = torch.tensor(power)
        power = torch.transpose(power.reshape((-1, 5)), 1, 0)
        power = power.type(torch.FloatTensor)

        if self.transform:
            power = self.transform(power)

        return inputs.permute(1, 0).float(), \
            (torch.tensor(state).long().squeeze(), power.float().squeeze())


def seq2pointnilm_get_datasets(data, load_train=True, load_test=True):

    (data_dir, args) = data

    transform = transforms.Compose([
                    ai8x.normalize(args=args),
                ])

    if load_train:
        train_dataset = Seq2PointNILM(root=data_dir, d_type='train', 
                                    transform=transform,
                                    seq_len = 100)
    else:
        train_dataset = None

    if load_test:
        test_dataset = Seq2PointNILM(root=data_dir, d_type='test',
                                    transform=transform,
                                    seq_len = 100)
    else:
        test_dataset = None

    return train_dataset, test_dataset

datasets = [
	{
		'name' : 'Seq2PointNILM',
		'input' : (1, 100),
		'output' : (0, 1, 2, 3, 4),
		'weights' : (1, 1),
		'loader' : seq2pointnilm_get_datasets,
	},
]

