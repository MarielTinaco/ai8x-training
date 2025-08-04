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
import os, sys
import contextlib
from pathlib import Path
from datetime import datetime
from functools import partial
from collections import namedtuple
from typing import Union, Iterable, Optional, Callable, Sequence

import torch
import numpy as np
import polars as pl
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.preprocessing import minmax_scale

import pandas as pd

import ai8x
from utils.nilm_utils import Sequence2Point, FixedRangeScaler, APPLIANCE_GLOBAL_DATA, chunked_numpy_processing


SITEMETER_KEY = "/site_meter/instance_1"

QUANTILE_FILTER_WINDOW = {
    "fridge freezer" : APPLIANCE_GLOBAL_DATA[0]["filter_window"],
    "kettle" : APPLIANCE_GLOBAL_DATA[1]["filter_window"],
    "washer dryer" : APPLIANCE_GLOBAL_DATA[2]["filter_window"],
    "dish washer" : APPLIANCE_GLOBAL_DATA[3]["filter_window"],
    "microwave" : APPLIANCE_GLOBAL_DATA[4]["filter_window"],
    "television" : APPLIANCE_GLOBAL_DATA[5]["filter_window"],
}

APPLIANCE_GLOBAL_MAX = {
    "fridge freezer" : APPLIANCE_GLOBAL_DATA[0]["max"],
    "kettle" : APPLIANCE_GLOBAL_DATA[1]["max"],
    "washer dryer" : APPLIANCE_GLOBAL_DATA[2]["max"],
    "dish washer" : APPLIANCE_GLOBAL_DATA[3]["max"],
    "microwave" : APPLIANCE_GLOBAL_DATA[4]["max"],
    "television" : APPLIANCE_GLOBAL_DATA[5]["max"],
}

class NILM(Dataset):

    class_dict = {'_noise_': 0, 'active_subwoofer': 1, 'audio_amplifier': 2, 'audio_system': 3, 'baby_monitor': 4, 'boiler': 5,
                  'bouncy_castle_pump': 6, 'breadmaker': 7, 'broadband_router': 8, 'charger': 9, 'clothes_iron': 10,
                  'coffee_maker': 11, 'computer': 12, 'computer_monitor': 13, 'desktop_computer': 14, 'dish washer': 15,
                  'drill': 16, 'ethernet_switch': 17, 'external_hard_disk': 18, 'fan': 19, 'food_processor': 20,
                  'fridge freezer': 21, 'hair_dryer': 22, 'hair_straighteners': 23, 'HTPC': 24, 'immersion_heater': 25,
                  'kettle': 26, 'kitchen_aid': 27, 'laptop_computer': 28, 'light': 29, 'microwave': 30, 'mobile_phone_charger': 31,
                  'oven': 32, 'printer': 33, 'radio': 34, 'security_alarm': 35, 'solar_thermal_pumping_station': 36,
                  'soldering_iron': 37, 'tablet_computer_charger': 38, 'television': 39, 'toasted_sandwich_maker': 40, 'toaster': 41,
                  'USB_hub': 42, 'vacuum_cleaner': 43, 'washer dryer': 44, 'water_pump': 45, 'wireless_phone_charger': 46}

    def __init__(self, root, filename, dtype, timeframe: tuple, classes : list, transform=None, seq_len=100,
                 synth_input=False, denoise_input=True, maximum_value=None, compand_input=False, high_precision_rms=False,
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
        self.maximum_value = maximum_value
        self.compand_input = compand_input
        self.high_precision_rms = high_precision_rms

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

        MetadataItem = namedtuple("MetadataItem", ["filepath", "metadata"])

        aggregate_metadata = []
        disaggregate_metadata = []

        # Orders source .parquet files into aggregate and disaggregates based on metadata
        for file in os.listdir(self.raw_folder):
            datafile = self.raw_folder / file
            metadata = pl.read_parquet_metadata(datafile)

            if metadata.get("is_aggregate") == "true":
                item = MetadataItem(filepath=datafile, metadata=metadata)
                aggregate_metadata.append(item)

            else:
                metadata["on_power_threshold"] = float(metadata["on_power_threshold"])     
                metadata["instance"] = int(metadata["instance"])

                # Get all instance 1's of the appliances
                if metadata["instance"] != 1:
                    continue

                item = MetadataItem(filepath=datafile, metadata=metadata)
                disaggregate_metadata.append(item)

        start_time = pl.datetime(self.timeframe[0].year, self.timeframe[0].month, self.timeframe[0].day)
        stop_time = pl.datetime(self.timeframe[1].year, self.timeframe[1].month, self.timeframe[1].day)

        # TODO: aggregate_metadata may contain multiple signals if the source is polyphase
        agg_info : MetadataItem = aggregate_metadata[0]

        agg_data = pl.read_parquet(agg_info.filepath)

        agg_data = agg_data.with_columns(
            pl.col("time").dt.replace_time_zone(None)
        )

        agg_data = agg_data.filter((agg_data["time"] >= start_time) & (agg_data["time"] <= stop_time))

        agg_data = agg_data.group_by_dynamic("time", every="6s").agg(pl.col("('power', 'active')").mean())

        agg_data = agg_data.rename({"('power', 'active')": "signal"})

        agg_synth_data = np.zeros(len(agg_data))
        rms_data = np.zeros((len(agg_data), len(self.classes)))
        states_data = np.zeros((len(agg_data), len(self.classes)))    

        for disagg in tqdm(disaggregate_metadata):
            # Load disaggregated signal from parquet file
            disagg_sample = pl.read_parquet(disagg.filepath)
            disagg_on_power_threshold = disagg.metadata.get("on_power_threshold")
            disagg_appliance_type = disagg.metadata.get("type")

            # Slice disaggregated signal
            disagg_sample = disagg_sample.with_columns(
                pl.col("time").dt.replace_time_zone(None)
            )

            disagg_sample = disagg_sample.filter((disagg_sample["time"] >= start_time) & (disagg_sample["time"] <= stop_time))

            # Match disaggregate data with the downsampled aggregated data
            matched_df = agg_data.join_asof(disagg_sample, on="time", strategy="nearest")
            disagg_matched = matched_df.select(["time", "('power', 'active')"])
            disagg_downsampled = disagg_matched.join(agg_data, on="time")
            disagg_data = disagg_downsampled.select(["time", "('power', 'active')"]).rename({"('power', 'active')": "signal"})

            # Align length of disaggregates to downsampled aggregated data 
            disagg_data = np.resize(disagg_data["signal"].to_numpy(), (len(agg_data),))

            # Clean data
            disagg_data = np.nan_to_num(disagg_data)

            # Aggregate data synthetically both target appliances and noise
            agg_synth_data += disagg_data

            # Define parameters for the quantile filter
            quantile_filter_by_sequence_length = partial(NILM.quantile_filter,
                                                         sequence_length=QUANTILE_FILTER_WINDOW[disagg_appliance_type])

            # Define parameters for the binarizer
            binarizer = lambda data: np.where(data >= disagg_on_power_threshold, 1, 0).astype(int)

            # scaler = MinMaxScaler(feature_range=(0, 1))
            scaler = FixedRangeScaler(min=0, max=APPLIANCE_GLOBAL_MAX[disagg_appliance_type], feature_range=(0, 1))

            # Parallel processing of large chunks of data
            data_filtered = chunked_numpy_processing(quantile_filter_by_sequence_length,
                                                     disagg_data,
                                                     chunk_size=100_000)

            data_binarized = chunked_numpy_processing(binarizer,
                                                      disagg_data,
                                                      chunk_size=100_000)

            data_norm = chunked_numpy_processing(scaler,
                                                 data_filtered,
                                                 chunk_size=100_000)

            if disagg_appliance_type in self.classes:
                # Fill by index of class to prevent collision
                class_idx = self.classes.index(disagg_appliance_type)
                rms_data[:, class_idx] = data_norm
                states_data[:, class_idx] = data_binarized

            del disagg_data
            del data_norm
            del data_binarized
            del data_filtered

        # Synthesize input or use site meter from dataset
        if self.synth_input:
            input_array = agg_synth_data
        else:
            input_array = agg_data["signal"].to_numpy()

        # Filter input data
        quantile_filter_by_16 = partial(NILM.quantile_filter, sequence_length=16)
        quantil_filter_by_seq_len = partial(NILM.quantile_filter, sequence_length=self.seq_len)
        mains_scaler = FixedRangeScaler(min=0,
                                        max=self.maximum_value if self.maximum_value else mains.max(),
                                        feature_range=(0, 1))

        # Filter data as is or apply noise then filter
        if self.denoise_input:
            mains = chunked_numpy_processing(quantil_filter_by_seq_len, mains, chunk_size=100_000)
        else:
            mains = input_array - np.percentile(input_array, 1)
            mains = np.where(mains < input_array, input_array, mains)
            mains = chunked_numpy_processing(quantile_filter_by_16, mains, chunk_size=100_000)

        # Normalize input
        mains = chunked_numpy_processing(mains_scaler, mains, chunk_size=100_000)

        if self.compand_input:
            mains = chunked_numpy_processing(NILM.mu_law_compand, mains, chunk_size=100_000)

        # mains = np.clip(mains, a_min=0, a_max=1)

        self.input_array = mains
        self.rms_array = rms_data
        self.states_array = states_data

        processed_data = self.input_array, (self.states_array, self.rms_array)

        Path(self.processed_folder / self.dtype).mkdir(exist_ok=True)
        torch.save(processed_data, self.processed_folder / self.dtype / self.data_file)

    # def __gen_datasets(self):

    #     with self.load_h5() as h5file:

    #         site_meter = h5file[SITEMETER_KEY].power.active
    #         site_meter = self.slice_by_datetime(site_meter, *self.timeframe)

    #         # Site meter resampled
    #         site_meter = site_meter.resample('6s').mean()

    #         # Filter instance 1 of selected classes
    #         # TODO: Support all classes and each instance
    #         class_active_power = self.filter_classes(h5file)

    #         # Slice each by specified date range
    #         sliced_per_date = map(lambda x: NILM.slice_by_datetime(x, *self.timeframe), class_active_power)

    #         # Sync datetime indices of appliance data to site meter and even out their lengths
    #         synced_datetime_to_site_meter = NILM.sync_datetime_index(site_meter, sliced_per_date)

    #         app_collection = []
    #         rms_list = []
    #         states_list = []

    #         for idx, app_df in enumerate(tqdm(synced_datetime_to_site_meter, total=len(self.classes))):
    #             device_info = h5file.get_storer(f"{self.classes[idx]}/instance_1").attrs.device_info

    #             on_power_threshold = device_info["on_power_threshold"]
    #             app_df = app_df.to_numpy().flatten()

    #             app_collection.append(app_df)

    #             filtered_df = NILM.quantile_filter(app_df, QUANTILE_FILTER_WINDOW[self.classes[idx]], p=50)
    #             # filtered_df = NILM.quantile_filter(app_df, self.seq_len, p=50)

    #             # normalized_df = minmax_scale(filtered_df, feature_range=(0, APPLIANCE_GLOBAL_MAX[self.classes[idx]]))
    #             normalized_df = minmax_scale(filtered_df)

    #             binarized_df = np.where(filtered_df >= on_power_threshold, 1, 0).astype(int)

    #             rms_list.append(normalized_df)
    #             states_list.append(binarized_df)

    #         # Synthesize input or use site meter from dataset
    #         if self.synth_input:
    #             input_array = np.sum(app_collection, axis=0)
    #         else:
    #             input_array = np.array(site_meter)

    #         # Filter data as is or apply noise then filter
    #         if self.denoise_input:
    #             mains = NILM.quantile_filter(input_array, self.seq_len)

    #         else:
    #             mains = input_array - np.percentile(input_array, 1)
    #             mains = np.where(mains < input_array, input_array, mains)
    #             mains = NILM.quantile_filter(mains, sequence_length=16, p=50)

    #         if self.maximum_value:
    #             mains = np.clip(mains, a_min=0, a_max=self.maximum_value)
    #             mains = minmax_scale(mains, feature_range=(0, self.maximum_value))
    #         else:
    #             mains = minmax_scale(mains)

    #         if self.compand_input:
    #             mains = NILM.mu_law_compand(mains)

    #         self.input_array = mains
    #         self.rms_array = np.vstack(rms_list).T
    #         self.states_array = np.vstack(states_list).T

    #         labels = self.states_array, self.rms_array
    #         processed_data = self.input_array, labels

    #         Path(self.processed_folder / self.dtype).mkdir(exist_ok=True)
    #         torch.save(processed_data, self.processed_folder / self.dtype / self.data_file)

    def select_loading_scheme(self, loading_scheme) -> Sequence:
        """
        Add other loading schemes here. Only add objects that follow Sequence Protocol here
        """
        if loading_scheme == "seq2point":
            return Sequence2Point(self.input_array,
                                  (self.states_array,self.rms_array),
                                  sequence_length = self.seq_len,
                                  stride=1,
                                  transform=self.transform,
                                  wide=bool(self.high_precision_rms))
        elif loading_scheme == "seq2point_stratified":
            return Sequence2Point(self.input_array,
                                  (self.states_array,self.rms_array),
                                  sequence_length = self.seq_len,
                                  stratified=True,
                                  transform=self.transform,
                                  wide=bool(self.high_precision_rms))
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
    def mu_law_compand(x, mu=255):
        return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)

    @staticmethod
    def mu_law_expand(y, mu=255):
        return np.sign(y) * (1 / mu) * (np.expm1(np.abs(y) * np.log1p(mu)))

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
    # classes = ["fridge freezer", "kettle", "washer dryer", "dish washer", "microwave",
    #            "television", "vacuum_cleaner", "toaster", "laptop_computer",
    #            "computer", "broadband_router", "charger"]
    classes = ["fridge freezer", "kettle", "washer dryer", "dish washer", "microwave"]
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
    # classes = ["fridge freezer", "kettle", "washer dryer", "dish washer", "microwave",
    #            "television", "vacuum_cleaner", "toaster", "laptop_computer",
    #            "computer", "broadband_router", "charger"]
    classes = ["fridge freezer", "kettle", "washer dryer", "dish washer", "microwave"]
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
    # classes = ["fridge freezer", "kettle", "washer dryer", "dish washer", "microwave",
    #            "television", "vacuum_cleaner", "toaster", "laptop_computer",
    #            "computer", "broadband_router", "charger"]
    classes = ["fridge freezer", "kettle", "washer dryer", "dish washer", "microwave"]
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
    # classes = ["fridge freezer", "kettle", "washer dryer", "dish washer", "microwave",
    #            "television", "vacuum_cleaner", "toaster", "laptop_computer",
    #            "computer", "broadband_router", "charger"]
    classes = ["fridge freezer", "kettle", "washer dryer", "dish washer", "microwave"]
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


def ukdale_128_seq2point_aug_stratified_get_datasets(data, load_train=True, load_test=True):

    UKDALE_SOURCE = "ukdale_bldg1_20121109_20170426.h5"
    AUG_UKDALE_SOURCE = "ukdale_bldg1_20121109_20170426_aug.h5"
    TRAIN_TIMEFRAME = datetime(year=2014, month=3, day=25), datetime(year=2014, month=8, day=27)
    TEST_TIMEFRAME = datetime(year=2015, month=4, day=27), datetime(year=2015, month=6, day=15)
    (data_dir, args) = data

    seq_len = 128
    # classes = ["fridge freezer", "kettle", "washer dryer", "dish washer", "microwave",
    #            "television", "vacuum_cleaner", "toaster", "laptop_computer",
    #            "computer", "broadband_router", "charger"]
    classes = ["fridge freezer", "kettle", "washer dryer", "dish washer", "microwave"]
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
                             loading_scheme="seq2point_stratified")
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


def ukdale_128_seq2point_stratified_get_datasets(data, load_train=True, load_test=True):

    UKDALE_SOURCE = "ukdale_bldg1_20121109_20170426.h5"
    TRAIN_TIMEFRAME = datetime(year=2014, month=3, day=25), datetime(year=2014, month=8, day=27)
    TEST_TIMEFRAME = datetime(year=2015, month=4, day=27), datetime(year=2015, month=7, day=30)
    MAXIMUM_VALUE = 4500
    (data_dir, args) = data

    seq_len = 128
    # classes = ["fridge freezer", "kettle", "washer dryer", "dish washer", "microwave",
    #            "television", "vacuum_cleaner", "toaster", "laptop_computer",
    #            "computer", "broadband_router", "charger"]
    classes = ["fridge freezer", "kettle", "washer dryer", "dish washer", "microwave"]
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
                             denoise_input=False,
                             maximum_value=MAXIMUM_VALUE,
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
                            denoise_input=False,
                            maximum_value=MAXIMUM_VALUE,
                            loading_scheme="seq2point")
    else:
        test_dataset = None

    return train_dataset, test_dataset


def ukdale_128_seq2point_stratified_crossval_get_datasets(data, load_train=True, load_test=True):

    AUG_UKDALE_SOURCE = "ukdale_bldg1_20140320_20150630_aug.h5"
    UKDALE_SOURCE = "ukdale_bldg1_20121109_20170426.h5"
    TRAIN_TIMEFRAME = datetime(year=2014, month=3, day=25), datetime(year=2014, month=8, day=27)
    TEST_TIMEFRAME = datetime(year=2015, month=2, day=27), datetime(year=2015, month=5, day=15)
    MAXIMUM_VALUE = 4500
    (data_dir, args) = data

    seq_len = 128
    # classes = ["fridge freezer", "kettle", "washer dryer", "dish washer", "microwave",
    #            "television", "vacuum_cleaner", "toaster", "laptop_computer",
    #            "computer", "broadband_router", "charger"]
    classes = ["fridge freezer", "kettle", "washer dryer", "dish washer", "microwave"]
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
                             denoise_input=False,
                             maximum_value=MAXIMUM_VALUE,
                             compand_input=True,
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
                            denoise_input=False,
                            maximum_value=MAXIMUM_VALUE,
                            compand_input=True,
                            loading_scheme="seq2point")
    else:
        test_dataset = None

    return train_dataset, test_dataset

def ukdale_128_seq2point_stratified_compand_get_datasets(data, load_train=True, load_test=True):

    UKDALE_SOURCE = "ukdale_bldg1_20121109_20170426.h5"
    TRAIN_TIMEFRAME = datetime(year=2014, month=3, day=25), datetime(year=2014, month=8, day=27)
    TEST_TIMEFRAME = datetime(year=2015, month=4, day=27), datetime(year=2015, month=7, day=30)
    MAXIMUM_VALUE = 4500
    (data_dir, args) = data

    seq_len = 128
    # classes = ["fridge freezer", "kettle", "washer dryer", "dish washer", "microwave",
    #            "television", "vacuum_cleaner", "toaster", "laptop_computer",
    #            "computer", "broadband_router", "charger"]
    classes = ["fridge freezer", "kettle", "washer dryer", "dish washer", "microwave"]
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
                             denoise_input=False,
                             maximum_value=MAXIMUM_VALUE,
                             compand_input=True,
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
                            denoise_input=False,
                            maximum_value=MAXIMUM_VALUE,
                            compand_input=True,
                            loading_scheme="seq2point")
    else:
        test_dataset = None

    return train_dataset, test_dataset


def ukdale_128_seq2point_stratified_compand_wide_get_datasets(data, load_train=True, load_test=True):

    UKDALE_SOURCE = "ukdale_bldg1_20121109_20170426.h5"
    TRAIN_TIMEFRAME = datetime(year=2014, month=3, day=25), datetime(year=2014, month=9, day=30)
    TEST_TIMEFRAME = datetime(year=2015, month=4, day=27), datetime(year=2015, month=7, day=30)
    MAXIMUM_VALUE = 4500
    (data_dir, args) = data

    seq_len = 128
    # classes = ["fridge freezer", "kettle", "washer dryer", "dish washer", "microwave",
    #            "television", "vacuum_cleaner", "toaster", "laptop_computer",
    #            "computer", "broadband_router", "charger"]
    classes = ["fridge freezer", "kettle", "washer dryer", "dish washer", "microwave"]
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
                             denoise_input=False,
                             maximum_value=MAXIMUM_VALUE,
                             compand_input=True,
                             high_precision_rms=True,
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
                            denoise_input=False,
                            maximum_value=MAXIMUM_VALUE,
                            compand_input=True,
                            high_precision_rms=True,
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
		'name' : 'UKDALE_128_stratified_aug',
		'input' : (1, 128),
		# 'output' : (21, 26, 44, 15, 30, 39, 43, 41, 28, 12, 8, 9),
		'output' : (21, 26, 44, 15, 30),
		'weight' : (1, 1),
		'loader' : ukdale_128_seq2point_aug_stratified_get_datasets,
	},
    {
		'name' : 'UKDALE_128_stratified_crossval',
		'input' : (1, 128),
		# 'output' : (21, 26, 44, 15, 30, 39, 43, 41, 28, 12, 8, 9),
		'output' : (21, 26, 44, 15, 30),
		'weight' : (1, 1),
		'loader' : ukdale_128_seq2point_stratified_crossval_get_datasets,
	},
    {
		'name' : 'UKDALE_128_stratified_compand',
		'input' : (1, 128),
		# 'output' : (21, 26, 44, 15, 30, 39, 43, 41, 28, 12, 8, 9),
		'output' : (21, 26, 44, 15, 30),
		'weight' : (1, 1),
		'loader' : ukdale_128_seq2point_stratified_compand_get_datasets,
	},
    {
		'name' : 'UKDALE_128_stratified_compand_wide',
		'input' : (1, 128),
		# 'output' : (21, 26, 44, 15, 30, 39, 43, 41, 28, 12, 8, 9),
		'output' : (21, 26, 44, 15, 30),
		'weight' : (1, 1),
		'loader' : ukdale_128_seq2point_stratified_compand_wide_get_datasets,
	}
]

if __name__ == "__main__":
    ...