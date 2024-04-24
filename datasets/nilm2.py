import numpy as np
import torch
import os
import functools

import torchvision
from torchvision import transforms

from pathlib import Path

import ai8x

DEFAULT_EXPERIMENT = 'unetnilm'

def split_data(data):
	split_1 = int(0.60 * len(data))
	split_2 = int(0.85 * len(data))
	train = data[:split_1]
	validation = data[split_1:split_2]
	test = data[split_2:]
	return train, validation, test

class NILM(torch.utils.data.Dataset):
    
	class_dict = {"fridge" : 0, "washer dryer" : 1, "kettle" : 2, "dish washer" : 3, "microwave" : 4}

	def __init__(self, root, classes, d_type, t_type="ukdale", transform=None, 
	      		 quantization_scheme=None, augmentation=None, download=False, 
				 save_unquantized=False, **kwargs):
	
		self.root = Path(root)
		self.classes = classes
		self.d_type = d_type
		self.t_type = t_type
		self.transform = transform
		self.save_unquantized = save_unquantized

		experiment = kwargs.get('experiment', DEFAULT_EXPERIMENT)
		denoise = kwargs.get('denoise', False)
		self.seq_len = kwargs.get('seq_len')
		self.data, self.targets = self.__load(experiment=experiment, denoise=denoise)
		self.indices = np.arange(self.data.shape[0])

	def __load(self,
	    	   experiment,
		   denoise,
		   data_type="training"):

		dirs = Path.iterdir(self.root / type(self).__name__ )
		dirs = filter(lambda x : x.name.split("_")[:2] == [experiment, self.t_type], dirs)
		latest_dir = max(dirs, key= lambda dir : dir.stat().st_mtime_ns)

		if denoise:
			x = np.load(latest_dir / data_type / "denoise_inputs.npy")
		else:
			x = np.load(latest_dir / data_type / "noise_inputs.npy")
		y = np.load(latest_dir / data_type / "targets.npy")
		z = np.load(latest_dir / data_type / "states.npy")

		return x, z

	def __len__(self):
		return self.data.shape[0] - self.seq_len

	def get_sample(self, index):
		indices = self.indices[index : index + self.seq_len]
		inds_inputs=sorted(indices[:self.seq_len])
		inds_targs=sorted(indices[self.seq_len-1:self.seq_len])

		return self.data[inds_inputs], self.targets[inds_targs]

	def __getitem__(self, index):
		inputs, state = self.get_sample(index)
		return torch.tensor(inputs).unsqueeze(-1).permute(1, 0).float(), torch.tensor(state).long().squeeze()


def ukdale_small_get_datasets(data, load_train=True, load_test=True):

	(data_dir, args) = data

	classes = ['fridge', 'washer dryer', 'kettle', 'dish washer', 'microwave']

	transform = transforms.Compose([
					ai8x.normalize(args=args)
    			])

	if load_train:
		train_dataset = NILM(root=data_dir, classes=classes, d_type='train', t_type='ukdale',
			      					transform=transform, download=False,
									seq_len = 100)
	else:
		train_dataset = None

	if load_test:
		test_dataset = NILM(root=data_dir, classes=classes, d_type='test', t_type='ukdale',
			      					transform=transform, download=False,
									seq_len = 100)
	else:
		test_dataset = None
	
	return train_dataset, test_dataset

datasets = [
	{
		'name' : 'UKDALE_small',
		'input' : (1, 100),
		'output' : (0, 1, 2, 3, 4),
		'weights' : (1, 1),
		'loader' : ukdale_small_get_datasets,
	}
]