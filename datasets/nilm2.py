import numpy as np
import torch
import os
import functools
import json

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

def split_data_2(data):
	split_1 = int(0.15 * len(data))
	test = data[:split_1]
	train = data[split_1:]
	return test, train


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
		self.metadata_path = None

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

		train_x, val_x, test_x = split_data(x)
		train_y, val_y, test_y = split_data(y)
		train_z, val_z, test_z = split_data(z)

		self.metadata_path = latest_dir / "metadata.json"

		if self.d_type == "train":
			x = train_x
			y = train_y
			z = train_z
		elif self.d_type == "test":
			x = test_x
			y = test_y
			z = test_z
		else:
			x = val_x
			y = val_y
			z = val_z

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

	@property
	def metadata(self):
		with open(self.metadata_path, "r") as infile:
                	return json.load(infile)

class NILMAutoEncoder(NILM):

	def __load(self,
	    	   experiment,
		   denoise,
		   data_type="training"):

		dirs = Path.iterdir(self.root / "NILM" )
		dirs = filter(lambda x : x.name.split("_")[:2] == [experiment, self.t_type], dirs)
		latest_dir = max(dirs, key= lambda dir : dir.stat().st_mtime_ns)

		if denoise:
			x = np.load(latest_dir / data_type / "denoise_inputs.npy")
		else:
			x = np.load(latest_dir / data_type / "noise_inputs.npy")
		y = np.load(latest_dir / data_type / "targets.npy")
		z = np.load(latest_dir / data_type / "states.npy")

		train_x, val_x, test_x = split_data(x)
		train_y, val_y, test_y = split_data(y)
		train_z, val_z, test_z = split_data(z)

		self.metadata_path = latest_dir / "metadata.json"

		if self.d_type == "train":
			x = train_x
			y = train_y
			z = train_z
		elif self.d_type == "test":
			x = test_x
			y = test_y
			z = test_z
		else:
			x = val_x
			y = val_y
			z = val_z

		return x, z


	def __getitem__(self, index):
		inputs, state = self.get_sample(index)
		return torch.tensor(inputs).unsqueeze(-1).float(), torch.tensor(state).long().squeeze()

	@property
	def metadata(self):
		with open(self.metadata_path, "r") as infile:
                	return json.load(infile)

class NILMRegress(torch.utils.data.Dataset):
    
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

		dirs = Path.iterdir(self.root / "NILM" )
		dirs = filter(lambda x : x.name.split("_")[:2] == [experiment, self.t_type], dirs)
		latest_dir = max(dirs, key= lambda dir : dir.stat().st_mtime_ns)
		if denoise:
			x = np.load(latest_dir / data_type / "denoise_inputs.npy")
		else:
			x = np.load(latest_dir / data_type / "noise_inputs.npy")
		y = np.load(latest_dir / data_type / "targets.npy")
		z = np.load(latest_dir / data_type / "states.npy")

		train_x, val_x, test_x = split_data(x)
		train_y, val_y, test_y = split_data(y)
		train_z, val_z, test_z = split_data(z)

		self.metadata_path = latest_dir / "metadata.json"

		if self.d_type == "train":
			x = train_x
			y = train_y
			z = train_z
		elif self.d_type == "test":
			x = test_x
			y = test_y
			z = test_z
		else:
			x = val_x
			y = val_y
			z = val_z

		return x, (z, y)

	def __len__(self):
		return self.data.shape[0] - self.seq_len

	def get_sample(self, index):
		indices = self.indices[index : index + self.seq_len]
		inds_inputs=sorted(indices[:self.seq_len])
		inds_targs=sorted(indices[self.seq_len-1:self.seq_len])
		
		states = self.targets[0]
		power = self.targets[1]

		return self.data[inds_inputs], (states[inds_targs], power[inds_targs])

	def __getitem__(self, index):
		inputs, targets = self.get_sample(index)
		state = targets[0]
		power = targets[1]
		return torch.tensor(inputs).unsqueeze(-1).permute(1, 0).float(), \
			(torch.tensor(state).long().squeeze(), torch.tensor(power).float().squeeze())

	@property
	def metadata(self):
		with open(self.metadata_path, "r") as infile:
                	return json.load(infile)

class NILMAutoEncoderRegress(torch.utils.data.Dataset):
    
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

		dirs = Path.iterdir(self.root / "NILM" )
		dirs = filter(lambda x : x.name.split("_")[:2] == [experiment, self.t_type], dirs)
		latest_dir = max(dirs, key= lambda dir : dir.stat().st_mtime_ns)

		if denoise:
			x = np.load(latest_dir / data_type / "denoise_inputs.npy")
		else:
			x = np.load(latest_dir / data_type / "noise_inputs.npy")
		y = np.load(latest_dir / data_type / "targets.npy")
		z = np.load(latest_dir / data_type / "states.npy")

		train_x, val_x, test_x = split_data(x)
		train_y, val_y, test_y = split_data(y)
		train_z, val_z, test_z = split_data(z)

		self.metadata_path = latest_dir / "metadata.json"

		if self.d_type == "train":
			x = train_x
			y = train_y
			z = train_z
		elif self.d_type == "test":
			x = test_x
			y = test_y
			z = test_z
		else:
			x = val_x
			y = val_y
			z = val_z

		return x, (z, y)

	def __len__(self):
		return self.data.shape[0] - self.seq_len

	def get_sample(self, index):
		indices = self.indices[index : index + self.seq_len]
		inds_inputs=sorted(indices[:self.seq_len])
		inds_targs=sorted(indices[self.seq_len-1:self.seq_len])
		
		states = self.targets[0]
		power = self.targets[1]

		return self.data[inds_inputs], (states[inds_targs], power[inds_targs])

	def __getitem__(self, index):
		inputs, targets = self.get_sample(index)
		state = targets[0]
		power = targets[1]
		return torch.tensor(inputs).unsqueeze(-1).float(), \
			(torch.tensor(state).long().squeeze(), torch.tensor(power).float().squeeze())

	@property
	def metadata(self):
		with open(self.metadata_path, "r") as infile:
                	return json.load(infile)


class NILMSlidingWindow(torch.utils.data.Dataset):

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
		self.height = kwargs.get('height')
		self.width = kwargs.get('width')

		self.stride = (self.seq_len - self.width) / self.height

		assert int(self.stride) == self.stride, f"({self.seq_len} - {self.width}) / {self.height} = {self.stride} stride must be an integer"

		self.stride = int(self.stride)

		self.data, self.targets = self.__load(experiment=experiment, denoise=denoise)
		self.indices = np.arange(self.data.shape[0])

	def __load(self,
	    	   experiment,
		   denoise,
		   data_type="training"):

		dirs = Path.iterdir(self.root / "NILM" )
		dirs = filter(lambda x : x.name.split("_")[:2] == [experiment, self.t_type], dirs)
		latest_dir = max(dirs, key= lambda dir : dir.stat().st_mtime_ns)

		if denoise:
			x = np.load(latest_dir / data_type / "denoise_inputs.npy")
		else:
			x = np.load(latest_dir / data_type / "noise_inputs.npy")
		y = np.load(latest_dir / data_type / "targets.npy")
		z = np.load(latest_dir / data_type / "states.npy")

		test_x, train_x = split_data_2(x)
		test_y, train_y = split_data_2(y)
		test_z, train_z = split_data_2(z)

		self.metadata_path = latest_dir / "metadata.json"

		if self.d_type == "train":
			x = train_x
			y = train_y
			z = train_z
		else:
			x = test_x
			y = test_y 
			z = test_z

		return x, (z, y)

	def __len__(self):
		return self.data.shape[0] - self.seq_len

	def get_sample(self, index):
		indices = self.indices[index : index + self.seq_len]
		inds_inputs=sorted(indices[:self.seq_len])
		inds_targs=sorted(indices[self.seq_len-1:self.seq_len])
		
		states = self.targets[0]
		power = self.targets[1]

		sequence = self.data[inds_inputs]
		sliding_window_data = np.zeros(shape=(self.height, self.width))
		for i in range(self.height):
			sliding_window_data[i, :] = sequence[self.stride * i : self.stride * i + self.width]

		return sliding_window_data, (states[inds_targs], power[inds_targs])


	def __getitem__(self, index):
		inputs, targets = self.get_sample(index)
		state = targets[0]
		power = targets[1]
		return torch.tensor(inputs).unsqueeze(0).float(), \
			(torch.tensor(state).long().squeeze(), torch.tensor(power).float().squeeze())

	@property
	def metadata(self):
		with open(self.metadata_path, "r") as infile:
                	return json.load(infile)

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


def ukdale_small_regress_get_datasets(data, load_train=True, load_test=True):

	(data_dir, args) = data

	classes = ['fridge', 'washer dryer', 'kettle', 'dish washer', 'microwave']

	transform = transforms.Compose([
					ai8x.normalize(args=args)
    			])

	if load_train:
		train_dataset = NILMRegress(root=data_dir, classes=classes, d_type='train', t_type='ukdale',
			      					transform=transform, download=False,
									seq_len = 100)
	else:
		train_dataset = None

	if load_test:
		test_dataset = NILMRegress(root=data_dir, classes=classes, d_type='test', t_type='ukdale',
			      					transform=transform, download=False,
									seq_len = 100)
	else:
		test_dataset = None

	return train_dataset, test_dataset

def ukdale_small_autoencoder_get_datasets(data, load_train=True, load_test=True):

	(data_dir, args) = data

	classes = ['fridge', 'washer dryer', 'kettle', 'dish washer', 'microwave']

	transform = transforms.Compose([
					ai8x.normalize(args=args)
    			])

	if load_train:
		train_dataset = NILMAutoEncoder(root=data_dir, classes=classes, d_type='train', t_type='ukdale',
			      					transform=transform, download=False,
									seq_len = 100)
	else:
		train_dataset = None

	if load_test:
		test_dataset = NILMAutoEncoder(root=data_dir, classes=classes, d_type='test', t_type='ukdale',
			      					transform=transform, download=False,
									seq_len = 100)
	else:
		test_dataset = None
	
	return train_dataset, test_dataset

def ukdale_small_autoencoder_regress_get_datasets(data, load_train=True, load_test=True):

	(data_dir, args) = data

	classes = ['fridge', 'washer dryer', 'kettle', 'dish washer', 'microwave']

	transform = transforms.Compose([
					ai8x.normalize(args=args)
    			])

	if load_train:
		train_dataset = NILMAutoEncoderRegress(root=data_dir, classes=classes, d_type='train', t_type='ukdale',
			      					transform=transform, download=False,
									seq_len = 100)
	else:
		train_dataset = None

	if load_test:
		test_dataset = NILMAutoEncoderRegress(root=data_dir, classes=classes, d_type='test', t_type='ukdale',
			      					transform=transform, download=False,
									seq_len = 100)
	else:
		test_dataset = None
	
	return train_dataset, test_dataset

def ukdale_small_sliding_window_get_datasets(data, load_train=True, load_test=True):

	(data_dir, args) = data

	classes = ['fridge', 'washer dryer', 'kettle', 'dish washer', 'microwave']

	transform = transforms.Compose([
					ai8x.normalize(args=args)
    			])

	if load_train:
		train_dataset = NILMSlidingWindow(root=data_dir, classes=classes, d_type='train', t_type='ukdale',
			      					transform=transform, download=False,
									seq_len = 100, width = 20,
									height = 20)
	else:
		train_dataset = None

	if load_test:
		test_dataset = NILMSlidingWindow(root=data_dir, classes=classes, d_type='test', t_type='ukdale',
			      					transform=transform, download=False,
									seq_len = 100, width=20,
									height= 20)
	else:
		test_dataset = None

	return train_dataset, test_dataset


def ukdale_small_512_regress_get_datasets(data, load_train=True, load_test=True):

	(data_dir, args) = data

	classes = ['fridge', 'washer dryer', 'kettle', 'dish washer', 'microwave']

	transform = transforms.Compose([
					ai8x.normalize(args=args)
    			])

	if load_train:
		train_dataset = NILMRegress(root=data_dir, classes=classes, d_type='train', t_type='ukdale',
			      					transform=transform, download=False,
									seq_len = 512)
	else:
		train_dataset = None

	if load_test:
		test_dataset = NILMRegress(root=data_dir, classes=classes, d_type='test', t_type='ukdale',
			      					transform=transform, download=False,
									seq_len = 512)
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
	},
	{
		'name' : 'UKDALE_small_regress',
		'input' : (1, 100),
		'output' : (0, 1, 2, 3, 4),
		'weights' : (1, 1),
		'loader' : ukdale_small_regress_get_datasets,
	},
	{
		'name' : 'UKDALE_small_autoencoder',
		'input' : (100, 1),
		'output' : (0, 1, 2, 3, 4),
		'weights' : (1, 1),
		'loader' : ukdale_small_autoencoder_get_datasets,
	},
	{
		'name' : 'UKDALE_small_autoencoder_regress',
		'input' : (100, 1),
		'output' : (0, 1, 2, 3, 4),
		'weights' : (1, 1),
		'loader' : ukdale_small_autoencoder_regress_get_datasets,
	},
	{
		'name' : 'UKDALE_small_sliding_window',
		'input' : (1, 20, 20),
		'output' : (0, 1, 2, 3, 4),
		'weights' : (1, 1),
		'loader' : ukdale_small_sliding_window_get_datasets,
	},
	{
		'name' : 'UKDALE_small_512_regress',
		'input' : (1, 512),
		'output' : (0, 1, 2, 3, 4),
		'weights' : (1, 1),
		'loader' : ukdale_small_512_regress_get_datasets,
	},
]