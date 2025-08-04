
from typing import Union, Iterable, Optional, Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import torch
import numpy as np
import torchnet.meter as tnt
from torch import nn
from sklearn.preprocessing import minmax_scale


APPLIANCE_GLOBAL_DATA = {
    "fridge freezer" : {
        "window": 50,
        "min": 0.0,
        "max": 259.0,
        "on_power_threshold": 50,
        "filter_window": 64
    },
    "kettle" : {
        "window": 50,
        "min": 0.0,
        "max": 2417.0,
        "on_power_threshold": 10,
        "filter_window": 8
    },
    "washer dryer" : {
        "window": 50,
        "min": 0.0,
        "max": 2055.0,
        "on_power_threshold": 20,
        "filter_window": 64
    },
    "dish washer" : {
        "window": 10,
        "min": 0.0,
        "max": 2439.0,
        "on_power_threshold": 10,
        "filter_window": 16
    },
    "microwave" : {
        "window": 50,
        "min": 0.0,
        "max": 1605.0,
        "on_power_threshold": 200,
        "filter_window": 32
    },
    "television" : {
        "window": 50,
        "min": 0.0,
        "max": 2500.0,
        "on_power_threshold": 200,
        "filter_window": 8
    }
}


class CustomNILMRegressionMetrics:

    def __init__(self, output_classes, *args, **kwargs):
        self.softmax = nn.Softmax(dim=1)
        self.mse_meter = tnt.MSEMeter()
        self.output_classes = output_classes
        self.num_classes = len(output_classes)

        self.reset()
        self.appliance_data = APPLIANCE_GLOBAL_DATA

    @staticmethod
    def _get_eac(target, prediction):
        num = np.abs(target - prediction).sum(axis=0)
        den = 2*target.sum(axis=0)
        return (1 - num/den)

    @staticmethod
    def _get_mae(target, prediction):
        return np.abs(target - prediction).mean(axis=0)

    @staticmethod
    def _get_nde(target, prediction):
        return np.sum((target - prediction) ** 2) / np.sum((target ** 2), axis=0) 

    @staticmethod
    def _compute_tp_fp_fn(true_targets, predictions, axis=1):
        # axis: axis for instance
        tp = np.sum(true_targets * predictions, axis=axis).astype('float32')
        fp = np.sum(np.logical_not(true_targets) * predictions,
                    axis=axis).astype('float32')
        fn = np.sum(true_targets * np.logical_not(predictions),
                    axis=axis).astype('float32')

        return (tp, fp, fn)

    @staticmethod
    def _example_f1_score(true_targets, predictions, per_sample=False, axis=1):
        tp, fp, fn = CustomNILMRegressionMetrics._compute_tp_fp_fn(true_targets, predictions, axis=axis)

        numerator = 2*tp
        denominator = (np.sum(true_targets,axis=axis).astype('float32') + np.sum(predictions,axis=axis).astype('float32'))

        zeros = np.where(denominator == 0)[0]

        denominator = np.delete(denominator,zeros)
        numerator = np.delete(numerator,zeros)

        example_f1 = numerator/denominator


        if per_sample:
            f1 = example_f1
        else:
            f1 = np.mean(example_f1)

        return f1

    @staticmethod
    def _compute_metrics(y_t, y_p):

        exf1_ = list(CustomNILMRegressionMetrics._example_f1_score(y_t, y_p, axis=0, per_sample=True))
        metrics_dict = {}
        metrics_dict['appF1'] = exf1_ 
        return metrics_dict

    @staticmethod
    def _compute_regress_metrics(y_t, y_p):
        eac = CustomNILMRegressionMetrics._get_eac(y_t, y_p)
        mae = CustomNILMRegressionMetrics._get_mae(y_t, y_p)
        nde = CustomNILMRegressionMetrics._get_nde(y_t, y_p)
        metrics_dict = {}
        metrics_dict['EAC'] = eac
        metrics_dict['MAE'] = mae
        metrics_dict['NDE'] = nde
        return metrics_dict

    def get_results_summary(self, z_t, z_p, y_t, y_p):

        reg = CustomNILMRegressionMetrics._compute_regress_metrics(y_t, y_p)
        mlb = CustomNILMRegressionMetrics._compute_metrics(z_t, z_p)

        per_app = {'EAC': reg['EAC'].tolist(),
                   'MAE': reg['MAE'].tolist(),
                   'NDE': reg['NDE'].tolist(),
                   'F1': mlb['appF1']}

        return per_app, {}

    def add(self, output, target):

        if not torch.is_tensor(output) and not torch.is_tensor(target):
            output = torch.from_numpy(output)
            target = torch.from_numpy(target)

        B = output.size(0)


        output_state = output[:,:2*5].reshape(B, 2, -1)
        output_power = output[:,2*5:].reshape(B, 5, -1)

        prob, pred_state = torch.max(self.softmax(output_state), 1)
        pred_power = torch.clip(output_power, min=-1)
        pred_power = (pred_power + 1) / 2
        
        y = target[1]
        z = target[0]

        logs = {"pred_power":pred_power, "pred_state":pred_state, "power":y, "state":z}
        self.outputs.append(logs)

        pred_rms = output_power[:,2]
        output = pred_state * pred_rms
        target = target[0] * target[1]
        self.mse_meter.add(output, target)

    def value(self):
        pred_power = torch.cat([x['pred_power'] for x in self.outputs], 0).cpu().numpy()
        pred_state = torch.cat([x['pred_state'] for x in self.outputs], 0).cpu().numpy().astype(np.int32)
        power = torch.cat([x['power'] for x in self.outputs], 0).cpu().numpy()
        state = torch.cat([x['state'] for x in self.outputs], 0).cpu().numpy().astype(np.int32)

        for idx, output_class in enumerate(self.output_classes):
            power[:,idx] = minmax_scale(power[:,idx], (APPLIANCE_GLOBAL_DATA[output_class]["min"], APPLIANCE_GLOBAL_DATA[output_class]["max"]))
            pred_power[:,:,idx] = np.clip(pred_power[:,:,idx], 0, 1)
            pred_power[:,:,idx] = minmax_scale(pred_power[:,:,idx], (APPLIANCE_GLOBAL_DATA[output_class]["min"], APPLIANCE_GLOBAL_DATA[output_class]["max"]))

        y_pred = pred_power[:,2]

        mse = self.mse_meter.value()

        per_app_results, avg_results = self.get_results_summary(state, pred_state, power, y_pred)

        return {
                "mse" : mse,
                'avg_eac' : np.mean(per_app_results['EAC']),
                'apps' : per_app_results,
        }

    def reset(self):
        self.outputs = []

        self.mse_meter.reset()


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
                 stratified = False,
                 transform = None,
                 wide=False):

        self.seq_len = sequence_length
        self.transform = transform
        self.wide = wide
        output_stride = lambda x: np.arange(sequence_length-1, labels[0].shape[0], stride)

        if stratified:
            stride = lambda _: self.activity_determined_indices(data, sequence_length=sequence_length)
            output_stride = lambda _: self.activity_determined_indices(data, sequence_length=sequence_length) + sequence_length

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

        if not self.wide:
            power_ = self.transform(power_)

        return self.transform(inp), \
			(torch.tensor(state).long().squeeze(), power_)


    def activity_determined_indices(self, activation_states, sequence_length):
        """

        """
        activity = activation_states

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


def chunked_numpy_processing(func, data_array, chunk_size=1000):
    """
    Processes a large NumPy array in chunks using ProcessPoolExecutor.
    """
    chunks = [data_array[i:i + chunk_size] for i in range(0, len(data_array), chunk_size)]

    with ThreadPoolExecutor() as executor:
        # Use executor.map to apply process_chunk to each chunk
        # chunksize argument can be tuned for performance
        results = list(executor.map(func, chunks, chunksize=4))

    # Concatenate the results back into a single NumPy array
    processed_array = np.concatenate(results)
    return processed_array


class FixedRangeScaler:

    def __init__(self, min, max, feature_range=(0, 1)):
        self.min = min
        self.max = max
        self.feature_range = feature_range
    
        assert self.min < self.max

    def __call__(self, data):
        data = (data - self.min) / (self.max - self.min)
        data = (self.feature_range[1] - self.feature_range[0])*data + self.feature_range[0]
        return data