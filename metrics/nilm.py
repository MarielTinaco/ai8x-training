
import torch
from torch import nn
import numpy as np
import torchnet.meter as tnt
from sklearn.preprocessing import minmax_scale


class CustomNILMRegressionMetrics:

    def __init__(self, num_classes, *args, **kwargs):
        self.softmax = nn.Softmax(dim=1)
        self.mse_meter = tnt.MSEMeter()
        self.num_classes = num_classes

        self.reset()
        self.appliance_data = [
            {
                "type": "fridge",
                "window": 50,
                "min": 0.0,
                "max": 259.0,
                "on_power_threshold": 50
            },
            {
                "type":"washer dryer",
                "window": 50,
                "min": 0.0,
                "max": 2055.0,
                "on_power_threshold": 20
            },
            {
                "type":"kettle",
                "window": 50,
                "min": 0.0,
                "max": 2417.0,
                "on_power_threshold": 10
            },
            {
                "type":"dish washer",
                "window": 10,
                "min": 0.0,
                "max": 2439.0,
                "on_power_threshold": 10
            },
            {
                "type":"microwave",
                "window": 50,
                "min": 0.0,
                "max": 1605.0,
                "on_power_threshold": 200
            }
        ]

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

        for idx, app_data in enumerate(self.appliance_data):
            power[:,idx] = minmax_scale(power[:,idx], (app_data["min"], app_data["max"]))
            pred_power[:,:,idx] = np.clip(pred_power[:,:,idx], 0, 1)
            pred_power[:,:,idx] = minmax_scale(pred_power[:,:,idx], (app_data["min"], app_data["max"]))

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