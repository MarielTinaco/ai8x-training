###################################################################################################
#
# Copyright (C) 2022-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
"""
import ast
import errno
import os
import pickle
import random
import sys
import contextlib
from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import Union, Iterable, Optional, Callable, Sequence
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import h5py
import pandas as pd

import ai8x


SITEMETER_KEY = "/site_meter/instance_1"

QUANTILE_FILTER_WINDOW = {
    "fridge_freezer" : 64,
    "kettle" : 1,
    "washer_dryer" : 64,
    "dish_washer" : 16,
    "microwave" : 64
}

APPLIANCE_GLOBAL_MAX = {
    "fridge_freezer" : 259.0,
    "kettle" : 2417.0,
    "washer_dryer" : 2055.0,
    "dish_washer" : 2439.0,
    "microwave" : 1605.0
}

class NILM(Dataset):

    class_dict = {'_noise_': 0, 'active_subwoofer': 1, 'audio_amplifier': 2, 'audio_system': 3, 'baby_monitor': 4, 'boiler': 5,
                  'bouncy_castle_pump': 6, 'breadmaker': 7, 'broadband_router': 8, 'charger': 9, 'clothes_iron': 10,
                  'coffee_maker': 11, 'computer': 12, 'computer_monitor': 13, 'desktop_computer': 14, 'dish_washer': 15,
                  'drill': 16, 'ethernet_switch': 17, 'external_hard_disk': 18, 'fan': 19, 'food_processor': 20,
                  'fridge_freezer': 21, 'hair_dryer': 22, 'hair_straighteners': 23, 'HTPC': 24, 'immersion_heater': 25,
                  'kettle': 26, 'kitchen_aid': 27, 'laptop_computer': 28, 'light': 29, 'microwave': 30, 'mobile_phone_charger': 31,
                  'oven': 32, 'printer': 33, 'radio': 34, 'security_alarm': 35, 'solar_thermal_pumping_station': 36,
                  'soldering_iron': 37, 'tablet_computer_charger': 38, 'television': 39, 'toasted_sandwich_maker': 40, 'toaster': 41,
                  'USB_hub': 42, 'vacuum_cleaner': 43, 'washer_dryer': 44, 'water_pump': 45, 'wireless_phone_charger': 46}

    def __init__(self, root, filename, dtype, timeframe: tuple, classes,
                 transform=None, seq_len=100, synth_input=False, denoise_input=True,
                 loading_scheme="seq2point"):

        if dtype not in ('test', 'train'):
            raise ValueError("dtype can only be set to 'test' or 'train'")

        self.root = Path(root)
        self.filename = filename
        self.dtype = dtype
        self.transform = transform
        self.timeframe = timeframe
        self.classes = classes
        self.seq_len = seq_len
        self.synth_input = synth_input
        self.denoise_input = denoise_input

        self.__makedir_exist_ok(self.processed_folder)

        self.input_array = None
        self.states_array = None
        self.rms_array = None

        self.data_file = "dataset.pt"

        if self.__check_exists():
            self.input_array, (self.states_array, self.rms_array) = \
                torch.load(os.path.join(self.processed_folder, self.dtype, self.data_file))
        else:
            self.__gen_datasets()

        self.loading_scheme = self.select_loading_scheme(loading_scheme)

    @property
    def raw_folder(self):
        """Folder for the raw data.
        """
        return Path(self.root) / "NILM" / "raw"

    @property
    def processed_folder(self):
        return Path(self.root) / "NILM" / "processed"

    def __len__(self):
        return len(self.loading_scheme)

    def __getitem__(self, index):
        return self.loading_scheme.__getitem__(index)

    @staticmethod
    def __reshape_audio(audio, row_len=128):
        # add overlap if necessary later on
        # return torch.transpose(audio.reshape((-1, row_len)), 1, 0)
        return audio.reshape((-1, row_len))

    @contextlib.contextmanager
    def load_h5(self):
        file : Path = self.raw_folder / self.filename
        if not file.exists():
            raise FileNotFoundError(file)

        with pd.HDFStore(file, mode="r", complevel=5) as store:
            yield store

    def __gen_datasets(self):

        with self.load_h5() as h5file:

            site_meter = h5file[SITEMETER_KEY].power.active
            site_meter = self.slice_by_datetime(site_meter, *self.timeframe)

            # Site meter resampled
            site_meter = site_meter.resample('6s').mean()

            # Filter instance 1 of selected classes
            # TODO: Support all classes and each instance
            class_active_power = self.filter_classes(h5file)

            # Slice each by specified date range
            sliced_per_date = map(lambda x: NILM.slice_by_datetime(x, *self.timeframe), class_active_power)

            # Sync datetime indices of appliance data to site meter and even out their lengths
            synced_datetime_to_site_meter = NILM.sync_datetime_index(site_meter, sliced_per_date)

            app_collection = []
            rms_list = []
            states_list = []

            for idx, app_df in enumerate(tqdm(synced_datetime_to_site_meter, total=len(self.classes))):
                device_info = h5file.get_storer(f"{self.classes[idx]}/instance_1").attrs.device_info

                on_power_threshold = device_info["on_power_threshold"]
                app_df = app_df.to_numpy().flatten()

                app_collection.append(app_df)

                filtered_df = NILM.quantile_filter(app_df, QUANTILE_FILTER_WINDOW[self.classes[idx]], p=50)
                # filtered_df = NILM.quantile_filter(app_df, self.seq_len, p=50)

                # normalized_df = minmax_scale(filtered_df, feature_range=(0, APPLIANCE_GLOBAL_MAX[self.classes[idx]]))
                normalized_df = minmax_scale(filtered_df)

                binarized_df = np.where(filtered_df >= on_power_threshold, 1, 0).astype(int)

                rms_list.append(normalized_df)
                states_list.append(binarized_df)

            # Synthesize input or use site meter from dataset
            if self.synth_input:
                input_array = np.sum(app_collection, axis=0)
            else:
                input_array = np.array(site_meter)

            # Filter data as is or apply noise then filter
            if self.denoise_input:
                mains = NILM.quantile_filter(input_array, self.seq_len)

            else:
                mains = input_array - np.percentile(input_array, 1)
                mains = np.where(mains < input_array, input_array, mains)
                mains = NILM.quantile_filter(mains, sequence_length=10, p=50)

            mains = minmax_scale(mains)

            self.input_array = mains
            self.rms_array = np.vstack(rms_list).T
            self.states_array = np.vstack(states_list).T

            labels = self.states_array, self.rms_array
            processed_data = self.input_array, labels

            Path(self.processed_folder / self.dtype).mkdir(exist_ok=True)
            torch.save(processed_data, self.processed_folder / self.dtype / self.data_file)

    def select_loading_scheme(self, loading_scheme) -> Sequence:
        """
        Add other loading schemes here. Only add objects that follow Sequence Protocol here
        """
        if loading_scheme == "seq2point":
            return Sequence2Point(self.input_array,
                                  (self.states_array,self.rms_array),
                                  sequence_length = self.seq_len,
                                  stride=1,
                                  transform=self.transform)
        elif loading_scheme == "seq2point_stratified":
            return Sequence2PointWithStratifiedSampling(self.input_array,
                                  (self.states_array,self.rms_array),
                                  sequence_length = self.seq_len,
                                  transform=self.transform)
        elif loading_scheme == "seq2point_stratified_on_input":
            return Sequence2PointWithStratifiedSampling(self.input_array,
                                  (self.states_array,self.rms_array),
                                  sequence_length = self.seq_len,
                                  transform=self.transform,                                  
                                  basis_vector="input")
        else:
            raise ValueError("Invalid Loading Scheme")


    def filter_classes(self, store: pd.HDFStore):
        for c in self.classes:
            instance_1 = store[f"{c}/instance_1"]
            active_power = instance_1.power.active

            yield active_power

    @staticmethod
    def scan(data, window_len):
        seq_len = window_len - 1 if window_len % 2==0 else window_len
        units_to_pad = seq_len // 2
        new_mains = np.pad(data, (units_to_pad,units_to_pad),'constant',constant_values=(0,0))  
        for i in range(len(new_mains) - seq_len+1):
            yield new_mains[i:i + seq_len]

    @staticmethod
    def quantile_filter(signal, sequence_length, p=50):
        new_signal = list(NILM.scan(np.array(signal), window_len=sequence_length))
        return np.percentile(new_signal, p, axis=1, method="nearest")

    @staticmethod
    def extract_available_appliances(store: pd.HDFStore):
        all_keys = store.keys()
        all_keys = (i.split("/")[1] for i in all_keys)
        all_keys = set(all_keys)
        all_keys = sorted(all_keys, key=lambda x: x.lower())
        all_keys.remove("site_meter")

        return all_keys

    @staticmethod
    def slice_by_datetime(df, starttime, endtime):
        datetime_index = pd.to_datetime(df.index).date
        date_slice = np.logical_and(datetime_index >= datetime.date(starttime), datetime_index <= datetime.date(endtime))
        return df[date_slice]

    @staticmethod
    def sync_datetime_index(df_base : pd.Series, dfs_to_sync : Iterable[pd.Series]):

        # Removes timezone from datetime index
        df_base_index = pd.to_datetime(df_base.index).tz_localize(None)
        
        df_base_ = pd.DataFrame(df_base_index, df_base, columns=["base"])
        
        for df_to_sync in dfs_to_sync:
            
            # Removes timezone from datetime index
            df_to_sync_index = pd.to_datetime(df_to_sync.index).tz_localize(None)
            df_to_sync_ = pd.DataFrame(df_to_sync_index, df_to_sync, columns=["to_sync"])
            # Syncs indexes
            merged = pd.merge_asof(df_base_.sort_values("base"),
                                   df_to_sync_.sort_values("to_sync"),
                                   left_on="base", right_on="to_sync",direction="nearest")

            # Selects the right column with sync-ed index from the merge
            merged_to_sync_index = merged["to_sync"]
            merged_to_sync_index = merged_to_sync_index.ffill()
            # Switches the index (active) and columns (datetime index)
            index_reset = df_to_sync_.reset_index()
            index_reset = index_reset.set_index("to_sync")

            # Creates a generator of active power indexed by datetime
            power_series_generator = (index_reset.loc[i] for i in merged_to_sync_index)

            # Converts it to series
            power_series = pd.DataFrame(power_series_generator, index=merged_to_sync_index)

            yield power_series


    @staticmethod
    def __makedir_exist_ok(dirpath):
        """Make directory if not already exists
        """
        Path(dirpath).mkdir(exist_ok=True)

    def __check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.dtype, self.data_file))


class WindowSampler:

    def __init__(self, data : np.ndarray,
                 stride : Optional[Union[int, np.ndarray, Callable]] = None, 
                 *args, **kwargs):
        """
        
        Parameters
        :param data:
        :type  data: numpy array

        Keyword arguments
        :param length: Length of sequence along axis
        :type  length: int
        :param axis: Axis of traversal
        :type  axis:
        """

        self._length = int(kwargs.get("length", 1))
        self._axis = int(kwargs.get("axis", 0))

        if len(data.shape) == 1:
            self._axis = 0

        indexable = data.shape[self._axis] - self._length

        if isinstance(stride, int):
            self.indices = np.arange(0, indexable, stride or 1)
        elif isinstance(stride, np.ndarray):
            self.indices = stride[stride < indexable]
        elif isinstance(stride, Callable):
            self.indices = stride(data)
            self.indices = self.indices[self.indices < indexable]
        else:
            self.indices = np.arange(0, indexable, 1)

        assert len(self.indices.shape) == 1, "Stride/Indexing must be 1D data"

        self.data = data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        ptr = self.indices[index]
        idx = [slice(None)]*self.data.ndim
        idx[self._axis] = range(ptr, ptr+self.length)
        return self.data[tuple(idx)]

    @property
    def length(self):
        return self._length

    @property
    def axis(self):
        return self._axis


# # # # # # # # # # # # # # # # # # # # # # #
#                                           #
#           DATA LOADING SCHEMES            #
#                                           #
# # # # # # # # # # # # # # # # # # # # # # #

class Sequence2Point(Sequence):
    """
    Implements Sequence Protocol
    """
    def __init__(self,
                 data: Union[np.ndarray, Iterable],
                 labels: Union[np.ndarray, Iterable],
                 sequence_length : int,
                 stride = 1,
                 transform = None):

        self.seq_len = sequence_length
        output_stride = lambda x: np.arange(sequence_length-1, labels[0].shape[0], stride)
        self.input_sampler = WindowSampler(data=data, length=sequence_length, axis=0, stride=stride)
        self.states_sampler = WindowSampler(data=labels[0], length=1, axis=0, stride= output_stride)
        self.rms_sampler = WindowSampler(data=labels[1], length=1, axis=0, stride= output_stride)
        self.transform = transform

    def __len__(self):
        return len(self.input_sampler)

    def __getitem__(self, index):
        if self.transform is None:
            return self.input_sampler[index], (self.states_sampler[index], self.rms_sampler[index])

        inputs = self.input_sampler[index]
        state = self.states_sampler[index]
        power = self.rms_sampler[index]

        # reshape to 2D
        inputs = torch.tensor(inputs)
        inp = inputs.reshape((-1, self.seq_len))
        inp = inp.type(torch.FloatTensor)

        power_ = torch.tensor(power)
        power_ = power_.type(torch.FloatTensor)
        power_ = power_.squeeze()

        return self.transform(inp), \
			(torch.tensor(state).long().squeeze(), self.transform(power_))


class Sequence2PointWithStratifiedSampling:

    def __init__(self,
                 data: Union[np.ndarray, Iterable],
                 labels: Union[np.ndarray, Iterable],
                 sequence_length : int,
                 transform=None,
                 basis_vector=None):

        self.seq_len = sequence_length
        self.transform = transform
        self.basis_vector = basis_vector
        if self.basis_vector == "input":
            stride = lambda _: self.activity_determined_indices(data, sequence_length=sequence_length)
            output_stride = lambda _: self.activity_determined_indices(data, sequence_length=sequence_length) + sequence_length
        else:
            stride = lambda _: self.activity_determined_indices(labels[0], sequence_length=sequence_length)
            output_stride = lambda _: self.activity_determined_indices(labels[0], sequence_length=sequence_length) + sequence_length

        self.input_sampler = WindowSampler(data=data, length=sequence_length, axis=0, stride=stride)
        self.states_sampler = WindowSampler(data=labels[0], length=1, axis=0, stride= output_stride)
        self.rms_sampler = WindowSampler(data=labels[1], length=1, axis=0, stride= output_stride)

    def __len__(self):
        return len(self.input_sampler)

    def __getitem__(self, index):
        if self.transform is None:
            return self.input_sampler[index], (self.states_sampler[index], self.rms_sampler[index])

        inputs = self.input_sampler[index]
        state = self.states_sampler[index]
        power = self.rms_sampler[index]

        # reshape to 2D
        inputs = torch.tensor(inputs)
        inp = inputs.reshape((-1, self.seq_len))
        inp = inp.type(torch.FloatTensor)

        power_ = torch.tensor(power)
        power_ = power_.type(torch.FloatTensor)
        power_ = power_.squeeze()

        return self.transform(inp), \
			(torch.tensor(state).long().squeeze(), self.transform(power_))

    def activity_determined_indices(self, activation_states, sequence_length):
        """

        """
        if self.basis_vector == "input":
            activity = activation_states
        else:
            # Aggregate activations states of each appliance for any given time to determing any activity
            activity = np.apply_along_axis(lambda x: int(any(x)), 1, activation_states)

        # Make sure there is only a single stream of data to determine activity
        assert len(activity.shape) == 1, "Aggregate activation state must be in 1 dimension to proceed"

        indices_with_detected_activity = set([])
        for idx, state in enumerate(activity):
            if state > 0:
                # Add an additional element to include full zero activation states
                start_index = idx - (sequence_length + 1)
                if start_index < 0:
                    continue
                
                limiter = idx + sequence_length
                if limiter >= len(activity):
                    break

                index_active = np.arange(max(0, start_index), idx, 1)
                indices_with_detected_activity.update(index_active)
        return np.sort(np.fromiter(indices_with_detected_activity, dtype=int))


# # # # # # # # # # # # # # # # # # # # #
#                                       #
#           DATASET GETTERS             #
#                                       #
# # # # # # # # # # # # # # # # # # # # #

def ukdale_seq2point_get_datasets(data, load_train=True, load_test=True):

    UKDALE_SOURCE = "ukdale_bldg1_20121109_20170426.h5"
    TRAIN_TIMEFRAME = datetime(year=2014, month=3, day=25), datetime(year=2014, month=8, day=27)
    TEST_TIMEFRAME = datetime(year=2015, month=4, day=27), datetime(year=2015, month=6, day=15)
    (data_dir, args) = data

    seq_len = 100
    # classes = ["fridge_freezer", "kettle", "washer_dryer", "dish_washer", "microwave",
    #            "television", "vacuum_cleaner", "toaster", "laptop_computer",
    #            "computer", "broadband_router", "charger"]
    classes = ["fridge_freezer", "kettle", "washer_dryer", "dish_washer", "microwave"]
    transform = transforms.Compose([ai8x.normalize(args=args)])

    if load_train:
        train_dataset = NILM(root=data_dir,
                             filename=UKDALE_SOURCE,
                             classes=classes,
                             dtype="train",
                             transform=transform,
                             timeframe=TRAIN_TIMEFRAME,
                             seq_len=seq_len,
                             synth_input=True,
                             denoise_input=True,
                             loading_scheme="seq2point")
    else:
        train_dataset = None

    if load_test:
        test_dataset = NILM(root=data_dir,
                            filename=UKDALE_SOURCE,
                            classes=classes,
                            dtype="test",
                            transform=transform,
                            timeframe=TEST_TIMEFRAME,
                            seq_len=seq_len,
                            synth_input=True,
                            denoise_input=True,
                            loading_scheme="seq2point")
    else:
        test_dataset = None

    return train_dataset, test_dataset

def ukdale_128_seq2point_get_datasets(data, load_train=True, load_test=True):

    UKDALE_SOURCE = "ukdale_bldg1_20121109_20170426_aug.h5"
    TRAIN_TIMEFRAME = datetime(year=2014, month=3, day=25), datetime(year=2014, month=8, day=27)
    TEST_TIMEFRAME = datetime(year=2015, month=4, day=27), datetime(year=2015, month=6, day=15)
    (data_dir, args) = data

    seq_len = 128
    # classes = ["fridge_freezer", "kettle", "washer_dryer", "dish_washer", "microwave",
    #            "television", "vacuum_cleaner", "toaster", "laptop_computer",
    #            "computer", "broadband_router", "charger"]
    classes = ["fridge_freezer", "kettle", "washer_dryer", "dish_washer", "microwave"]
    transform = transforms.Compose([ai8x.normalize(args=args)])

    if load_train:
        train_dataset = NILM(root=data_dir,
                             filename=UKDALE_SOURCE,
                             classes=classes,
                             dtype="train",
                             transform=transform,
                             timeframe=TRAIN_TIMEFRAME,
                             seq_len=seq_len,
                             synth_input=True,
                             denoise_input=True,
                             loading_scheme="seq2point")
    else:
        train_dataset = None

    if load_test:
        test_dataset = NILM(root=data_dir,
                            filename=UKDALE_SOURCE,
                            classes=classes,
                            dtype="test",
                            transform=transform,
                            timeframe=TEST_TIMEFRAME,
                            seq_len=seq_len,
                            synth_input=True,
                            denoise_input=True,
                            loading_scheme="seq2point")
    else:
        test_dataset = None

    return train_dataset, test_dataset


def ukdale_seq2point_stratified_get_datasets(data, load_train=True, load_test=True):

    UKDALE_SOURCE = "ukdale_bldg1_20121109_20170426.h5"
    TRAIN_TIMEFRAME = datetime(year=2013, month=3, day=25), datetime(year=2013, month=7, day=27)
    TEST_TIMEFRAME = datetime(year=2014, month=4, day=27), datetime(year=2014, month=5, day=28)
    (data_dir, args) = data

    seq_len = 100
    # classes = ["fridge_freezer", "kettle", "washer_dryer", "dish_washer", "microwave",
    #            "television", "vacuum_cleaner", "toaster", "laptop_computer",
    #            "computer", "broadband_router", "charger"]
    classes = ["fridge_freezer", "kettle", "washer_dryer", "dish_washer", "microwave"]
    transform = transforms.Compose([ai8x.normalize(args=args)])

    if load_train:
        train_dataset = NILM(root=data_dir,
                             filename=UKDALE_SOURCE,
                             classes=classes,
                             dtype="train",
                             transform=transform,
                             timeframe=TRAIN_TIMEFRAME,
                             seq_len=seq_len,
                             synth_input=True,
                             denoise_input=True,
                             loading_scheme="seq2point_stratified")
    else:
        train_dataset = None

    if load_test:
        test_dataset = NILM(root=data_dir,
                            filename=UKDALE_SOURCE,
                            classes=classes,
                            dtype="test",
                            transform=transform,
                            timeframe=TEST_TIMEFRAME,
                            seq_len=seq_len,
                            synth_input=True,
                            denoise_input=True,
                            loading_scheme="seq2point_stratified")
    else:
        test_dataset = None

    return train_dataset, test_dataset

def ukdale_seq2point_stratified_get_datasets(data, load_train=True, load_test=True):

    UKDALE_SOURCE = "ukdale_bldg1_20121109_20170426.h5"
    TRAIN_TIMEFRAME = datetime(year=2013, month=3, day=25), datetime(year=2013, month=7, day=27)
    TEST_TIMEFRAME = datetime(year=2014, month=4, day=27), datetime(year=2014, month=5, day=28)
    (data_dir, args) = data

    seq_len = 100
    # classes = ["fridge_freezer", "kettle", "washer_dryer", "dish_washer", "microwave",
    #            "television", "vacuum_cleaner", "toaster", "laptop_computer",
    #            "computer", "broadband_router", "charger"]
    classes = ["fridge_freezer", "kettle", "washer_dryer", "dish_washer", "microwave"]
    transform = transforms.Compose([ai8x.normalize(args=args)])

    if load_train:
        train_dataset = NILM(root=data_dir,
                             filename=UKDALE_SOURCE,
                             classes=classes,
                             dtype="train",
                             transform=transform,
                             timeframe=TRAIN_TIMEFRAME,
                             seq_len=seq_len,
                             synth_input=True,
                             denoise_input=True,
                             loading_scheme="seq2point_stratified_on_input")
    else:
        train_dataset = None

    if load_test:
        test_dataset = NILM(root=data_dir,
                            filename=UKDALE_SOURCE,
                            classes=classes,
                            dtype="test",
                            transform=transform,
                            timeframe=TEST_TIMEFRAME,
                            seq_len=seq_len,
                            synth_input=True,
                            denoise_input=True,
                            loading_scheme="seq2point_stratified_on_input")
    else:
        test_dataset = None

    return train_dataset, test_dataset


def ukdale_128_seq2point_stratified_get_datasets(data, load_train=True, load_test=True):

    UKDALE_SOURCE = "ukdale_bldg1_20121109_20170426.h5"
    AUG_UKDALE_SOURCE = "ukdale_bldg1_20121109_20170426_aug.h5"
    TRAIN_TIMEFRAME = datetime(year=2014, month=3, day=25), datetime(year=2014, month=8, day=27)
    TEST_TIMEFRAME = datetime(year=2015, month=4, day=27), datetime(year=2015, month=6, day=15)
    (data_dir, args) = data

    seq_len = 128
    # classes = ["fridge_freezer", "kettle", "washer_dryer", "dish_washer", "microwave",
    #            "television", "vacuum_cleaner", "toaster", "laptop_computer",
    #            "computer", "broadband_router", "charger"]
    classes = ["fridge_freezer", "kettle", "washer_dryer", "dish_washer", "microwave"]
    transform = transforms.Compose([ai8x.normalize(args=args)])

    if load_train:
        train_dataset = NILM(root=data_dir,
                             filename=AUG_UKDALE_SOURCE,
                             classes=classes,
                             dtype="train",
                             transform=transform,
                             timeframe=TRAIN_TIMEFRAME,
                             seq_len=seq_len,
                             synth_input=True,
                             denoise_input=True,
                             loading_scheme="seq2point_stratified_on_input")
    else:
        train_dataset = None

    if load_test:
        test_dataset = NILM(root=data_dir,
                            filename=AUG_UKDALE_SOURCE,
                            classes=classes,
                            dtype="test",
                            transform=transform,
                            timeframe=TEST_TIMEFRAME,
                            seq_len=seq_len,
                            synth_input=True,
                            denoise_input=True,
                            loading_scheme="seq2point")
    else:
        test_dataset = None

    return train_dataset, test_dataset

def ukdale_128_seq2point_stratified_crossval_get_datasets(data, load_train=True, load_test=True):

    AUG_UKDALE_SOURCE = "ukdale_bldg1_20140320_20150630_aug.h5"
    UKDALE_SOURCE = "ukdale_bldg1_20121109_20170426.h5"
    TRAIN_TIMEFRAME = datetime(year=2014, month=3, day=25), datetime(year=2014, month=8, day=27)
    TEST_TIMEFRAME = datetime(year=2015, month=3, day=27), datetime(year=2015, month=6, day=15)
    (data_dir, args) = data

    seq_len = 128
    # classes = ["fridge_freezer", "kettle", "washer_dryer", "dish_washer", "microwave",
    #            "television", "vacuum_cleaner", "toaster", "laptop_computer",
    #            "computer", "broadband_router", "charger"]
    classes = ["fridge_freezer", "kettle", "washer_dryer", "dish_washer", "microwave"]
    transform = transforms.Compose([ai8x.normalize(args=args)])

    if load_train:
        train_dataset = NILM(root=data_dir,
                             filename=AUG_UKDALE_SOURCE,
                             classes=classes,
                             dtype="train",
                             transform=transform,
                             timeframe=TRAIN_TIMEFRAME,
                             seq_len=seq_len,
                             synth_input=True,
                             denoise_input=True,
                             loading_scheme="seq2point_stratified_on_input")
    else:
        train_dataset = None

    if load_test:
        test_dataset = NILM(root=data_dir,
                            filename=UKDALE_SOURCE,
                            classes=classes,
                            dtype="test",
                            transform=transform,
                            timeframe=TEST_TIMEFRAME,
                            seq_len=seq_len,
                            synth_input=True,
                            denoise_input=True,
                            loading_scheme="seq2point")
    else:
        test_dataset = None

    return train_dataset, test_dataset



datasets = [
	{
		'name' : 'UKDALE',
		'input' : (1, 100),
		# 'output' : (21, 26, 44, 15, 30, 39, 43, 41, 28, 12, 8, 9),
		'output' : (21, 26, 44, 15, 30),
		'weight' : (0.00625, 1),
		'loader' : ukdale_seq2point_get_datasets,
	},
    {
		'name' : 'UKDALE_128',
		'input' : (1, 128),
		# 'output' : (21, 26, 44, 15, 30, 39, 43, 41, 28, 12, 8, 9),
		'output' : (21, 26, 44, 15, 30),
		'weight' : (0.00625, 1),
		'loader' : ukdale_128_seq2point_get_datasets,
	},
    	{
		'name' : 'UKDALE_stratified',
		'input' : (1, 100),
		# 'output' : (21, 26, 44, 15, 30, 39, 43, 41, 28, 12, 8, 9),
		'output' : (21, 26, 44, 15, 30),
		'weight' : (0.0625, 1),
		'loader' : ukdale_seq2point_stratified_get_datasets,
	},
    {
		'name' : 'UKDALE_stratified_on_input',
		'input' : (1, 100),
		# 'output' : (21, 26, 44, 15, 30, 39, 43, 41, 28, 12, 8, 9),
		'output' : (21, 26, 44, 15, 30),
		'weight' : (0.0625, 1),
		'loader' : ukdale_seq2point_stratified_get_datasets,
	},
    {
		'name' : 'UKDALE_128_stratified',
		'input' : (1, 128),
		# 'output' : (21, 26, 44, 15, 30, 39, 43, 41, 28, 12, 8, 9),
		'output' : (21, 26, 44, 15, 30),
		'weight' : (1, 1),
		'loader' : ukdale_128_seq2point_stratified_get_datasets,
	},
    {
		'name' : 'UKDALE_128_stratified_crossval',
		'input' : (1, 128),
		# 'output' : (21, 26, 44, 15, 30, 39, 43, 41, 28, 12, 8, 9),
		'output' : (21, 26, 44, 15, 30),
		'weight' : (1, 1),
		'loader' : ukdale_128_seq2point_stratified_crossval_get_datasets,
	}
]