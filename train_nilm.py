


import os
import sys
import datetime
import time

import importlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import operator

from torch import nn
from pathlib import Path
from collections import OrderedDict

import distiller

try:
    import tensorboard  # pylint: disable=import-error
    import tensorflow  # pylint: disable=import-error
    tensorflow.io.gfile = tensorboard.compat.tensorflow_stub.io.gfile
except (ModuleNotFoundError, AttributeError):
    pass

import torchnet.meter as tnt
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import distiller.apputils as apputils
from distiller.data_loggers import PythonLogger, TensorBoardLogger

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'models'))
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'datasets'))

from datasets import nilm
import ai8x
mod = importlib.import_module("models.ai87net-unetnilm")

msglogger = None

from torchmetrics.functional import f1_score

from unetnilm.metrics import get_results_summary
from unetnilm.utils import QuantileLoss

dataset_name = "ukdale"
dataset_fn = nilm.ukdale_get_datasets
model_name = "cnn1dnilm"
num_classes = 5
workers = 5
batch_size = 256
data_path = "data/NILM/"
deterministic = True
log_prefix = "ukdale-train"
log_dir = "logs"
validation_split = 0.1
seq_len = 99
print_freq = 10
num_epochs = 50
dropout=0.25
pool_filter = 8
lr = 1e-4
beta_1 = 0.999
beta_2 = 0.98
quantiles = [0.0025,0.1, 0.5, 0.9, 0.975]
patience_scheduler = 5
qat_policy = {'start_epoch':10,
              'weight_bits':8}
appliance_data = {
    "kettle": {
        "mean": 700,
        "std": 1000,
        'window':10,
        'on_power_threshold': 2000,
        'max_on_power': 3998
    },
    "fridge": {
        "mean": 200,
        "std": 400,
        "window":50,
        'on_power_threshold': 50,
    },
    "dish washer": {
        "mean": 700,
        "std": 700,
        "window":50,
        'on_power_threshold': 10
    },
    "washer dryer": {
        "mean": 400,
        "std": 700,
        "window":50,
        'on_power_threshold': 20,
        'max_on_power': 3999
    },
    "microwave": {
        "mean": 500,
        "std": 800,
        "window":10,
        'on_power_threshold': 200,
    },
}
appliances = list(appliance_data.keys())

#####################################################################################

class Args:
        def __init__(self, act_mode_8bit):
                self.act_mode_8bit = act_mode_8bit
                self.truncate_testset = False

def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

####################################################################################

msglogger = apputils.config_pylogger('logging.conf', log_prefix,
                                        log_dir)

pylogger = PythonLogger(msglogger, log_1d=True)
all_loggers = [pylogger]

# tensorboard
tflogger = TensorBoardLogger(msglogger.logdir, log_1d=True, comment='_'+dataset_name)

tflogger.tblogger.writer.add_text('Command line', "args ---")

msglogger.info('dataset_name:%s\ndataset_fn=%s\nnum_classes=%d\nmodel_name=%s\seq_len=%s\nbatch_size=%d\nvalidation_split=%s\nlr=%f',
                dataset_name,dataset_fn,num_classes,model_name,seq_len,batch_size,validation_split,lr)

##########################################################################################

args = Args(act_mode_8bit=False)

class NormDen:
        def __init__(self, mini, maxi):
               self.mini = mini
               self.maxi = maxi

        # def normalize(self, data):
        #         data = (data - self.mini) / (self.maxi - self.mini)
        #         return data.sub(0.5).mul(256.).round().clamp(min=-128, max=127).div(128.)

        def normalize(self, data):
                # Data must be [self.mini,self.maxi]
                data = (data - self.mini) / (self.maxi - self.mini)
                data[data > 1] = 1
                data[data < 0] = 0
                return data.sub(0.5).mul(256.).round().clamp(min=-127, max=128).div(128.)

        def denormalize(self, data):
                # Data must be [-1,1]
                drange = self.maxi - self.mini
                data = (data/2 + 0.5) * drange
                data = data + self.mini
                return data
        
        
########################################################################################

train_set, test_set, val_set = dataset_fn((data_path, args), load_train=True, load_test=True, load_val=True)

############################################################################################

train_loader, val_loader, test_loader, _ = apputils.get_data_loaders(
        dataset_fn, (data_path,args), batch_size,
        workers, validation_split, deterministic,1, 1, 1)
msglogger.info('Dataset sizes:\n\ttraining=%d\n\tvalidation=%d\n\ttest=%d',
                   len(train_loader.sampler), len(val_loader.sampler), len(test_loader.sampler))
msglogger.info('Augmentations:%s',train_loader.dataset.transform)

#################################################################################################

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

###################################################################################################

ai8x.set_device(device=85, simulate=False, round_avg=False)

model = mod.AI85CNN1DNiLM(in_size=1, output_size=5, n_layers=5, d_model=128, n_quantiles=len(quantiles), dropout=dropout, pool_filter=pool_filter)
msglogger.info('model: %s',model)
model = model.to(device)

msglogger.info('Number of Model Params: %d',count_params(model))

# configure tensorboard
# dummy_input = torch.randn(1, seq_len)
# dummy_input = batch_select[0]
# tflogger.tblogger.writer.add_graph(model.to('cpu'), (dummy_input, ), False)

# model = model.to(device)

all_loggers.append(tflogger)
all_tbloggers = [tflogger]

########################################################################################################

msglogger.info('epochs: %d',num_epochs)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta_1, beta_2))
msglogger.info('Optimizer Type: %s', type(optimizer))
# ms_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 35,100], gamma=0.5)
# msglogger.info("lr_schedule:%s","base: "+str(ms_lr_scheduler.base_lrs)+" milestones: "+str(ms_lr_scheduler.milestones)+ " gamma: "+str(ms_lr_scheduler.gamma))
ms_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience_scheduler, verbose=True, min_lr=1e-6, mode="min")
# criterion = torch.nn.CrossEntropyLoss().to(device)

criterion = QuantileLoss(quantiles=quantiles).to(device)

msglogger.info('qat policy: %s',qat_policy)
compression_scheduler = distiller.CompressionScheduler(model)

############################################################################################################

def test_epoch_end(outputs):
        
        # appliance_data = {'fridge': {'window': 50, 'mean': 40.158577, 'std': 53.56288}, 'washer dryer': {'window': 50, 'mean': 27.768433, 'std': 212.51971}, 'kettle': {'window': 10, 'mean': 16.753872, 'std': 191.05873}, 'dish washer': {'window': 50, 'mean': 27.384077, 'std': 239.23492}, 'microwave': {'window': 10, 'mean': 8.35921, 'std': 105.1099}}
        pred_power = torch.cat([x['pred_power'] for x in outputs], 0).cpu().numpy()
        pred_state = torch.cat([x['pred_state'] for x in outputs], 0).cpu().numpy().astype(np.int32)
        power = torch.cat([x['power'] for x in outputs], 0).cpu().numpy()
        state = torch.cat([x['state'] for x in outputs], 0).cpu().numpy().astype(np.int32)

        for idx, app in enumerate(appliances):
                power[:,idx] = (power[:, idx] * appliance_data[app]['std']) + appliance_data[app]['mean']
                if len(quantiles)>=2:
                        pred_power[:,:, idx] = (pred_power[:,:, idx] * appliance_data[app]['std']) + appliance_data[app]['mean']
                        pred_power[:,:, idx] = np.where(pred_power[:,:, idx]<0, 0, pred_power[:,:, idx])
                else:
                        pred_power[:, idx] = (pred_power[:, idx] * appliance_data[app]['std']) + appliance_data[app]['mean']
                        pred_power[:, idx] = np.where(pred_power[:, idx]<0, 0, pred_power[:, idx])    

        if len(quantiles)>=2:
                idx = len(quantiles)//2
                y_pred = pred_power[:,idx]
        else:
                y_pred = pred_power 
                
        per_app_results, avg_results = get_results_summary(state, pred_state, 
                                                                        power, y_pred,
                                                                        appliances, 
                                                                        dataset_name.upper())  
        logs = {"pred_power":pred_power, 
                "pred_state":pred_state, 
                "power":power, 
                "state":state,  
                'app_results':per_app_results, 
                'avg_results':avg_results} 
        
        return logs   

def validate(data_loader, model, criterion, loggers, epoch=-1, tflogger=None):
        """Execute the validation/test loop."""

        # store loss stats
        losses = {'objective_loss': tnt.AverageValueMeter()}
        classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, min(num_classes, 5)))

        # validation set info
        batch_time = tnt.AverageValueMeter()
        total_samples = len(data_loader.sampler)
        batch_size = data_loader.batch_size
        confusion = tnt.ConfusionMeter(num_classes)
        total_steps = (total_samples + batch_size - 1) // batch_size
        msglogger.info('%d samples (%d per mini-batch)', total_samples, batch_size)

        outputs = []

        # Switch to evaluation mode
        model.eval()

        # iterate over the batches in the validation set
        for validation_step, (inputs, target, states) in enumerate(data_loader):
                with torch.no_grad():

                        inputs, target, states = inputs.to(device), target.to(device), states.to(device)
                        # compute output from model

                        # inputs = normden.normalize(inputs)

                        # forward pass and loss calculation
                        logits, rmse_logits = model(inputs)

                        # rmse_logits = normden.denormalize(rmse_logits)
                        # logits = normden.denormalize(logits)

                        prob, pred = torch.max(F.softmax(logits, 1), 1)
                        loss_nll   = F.nll_loss(F.log_softmax(logits, 1), states)

                        if len(quantiles)>1:
                                prob=prob.unsqueeze(1).expand_as(rmse_logits)
                                loss_mse = criterion(rmse_logits, target)
                                mae_score = F.l1_loss(rmse_logits,target.unsqueeze(1).expand_as(rmse_logits))
                        else:    
                                loss_mse = F.mse_loss(rmse_logits, target)
                                mae_score = F.l1_loss(rmse_logits, target)

                        # UNETNILM
                        loss = loss_nll + loss_mse
                        res = f1_score(pred, states, task="multiclass", num_classes=5)
                        logs = {"nlloss":loss_nll, "mseloss":loss_mse,
                                "mae":mae_score, "F1": res}

                        losses['objective_loss'].add(loss.item())

                        steps_completed = (validation_step+1)
                        if steps_completed % print_freq == 0 or steps_completed == total_steps:
                                stats = ('Performance/Validation/',
                                        OrderedDict([('Loss', losses['objective_loss'].mean),('Time', batch_time.mean)]))
                                params = None
                                distiller.log_training_progress(stats,
                                                                params,
                                                                epoch, steps_completed,
                                                                steps_per_epoch, print_freq,
                                                                all_loggers)

                        outputs.append(logs)
        
        # OPTIONAL
        avg_loss = np.mean([x['nlloss'].item()+x['mseloss'].item() for x in outputs])
        avg_f1 = np.mean([x['F1'].item() for x in outputs])
        avg_rmse = np.mean([x['mae'].item() for x in outputs])
        logs = {'val_loss': avg_loss, "val_F1": avg_f1, "val_mae":avg_rmse}
        
        return {'log':logs}
        

if __name__ == "__main__":
        ####################################################################################

        # store model history across epochs
        perf_scores_history = []

        name = model_name

        # start the clock
        tic = datetime.datetime.now()

        normden = NormDen(mini=-1.0, maxi=12.92)

        # training loop
        for epoch in range(num_epochs):

                if epoch > 0 and epoch == qat_policy['start_epoch']:
                        print('QAT is starting!')
                        # Fuse the BN parameters into conv layers before Quantization 
                        ai8x.fuse_bn_layers(model)
                
                        # Switch model from unquantized to quantized for QAT
                        ai8x.initiate_qat(model, qat_policy)

                        # Model is re-transferred to GPU in case parameters were added
                        model.to(device)

                        # Empty the performance scores list for QAT operation
                        perf_scores_history = []
                        name = f'{model_name}_qat'
                
                # store loss and training stats
                losses = {'objective_loss': tnt.AverageValueMeter()}
                classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, min(num_classes, 5)))
                batch_time = tnt.AverageValueMeter()
                data_time = tnt.AverageValueMeter()

                # logging stats
                total_samples = len(train_loader.sampler)
                batch_size = train_loader.batch_size
                steps_per_epoch = (total_samples + batch_size - 1) // batch_size
                msglogger.info('Training epoch: %d samples (%d per mini-batch)', total_samples, batch_size)

                # Switch to train mode
                model.train()
                acc_stats = []
                end = time.time()

                outputs = []

                # iterate over all batches in the dataset    
                for train_step, (inputs, target, states) in enumerate(train_loader):
                        torch.cuda.empty_cache()
                        # Measure data loading time
                        data_time.add(time.time() - end)

                        inputs, target, states = inputs.to(device), target.to(device), states.to(device)

                        B = inputs.size(0)

                        # inputs = normden.normalize(inputs)

                        # forward pass and loss calculation
                        logits, rmse_logits = model(inputs)

                        # print(f"RMSE LOGITS:\t{rmse_logits.mean()} {rmse_logits.max()} {rmse_logits.min()}")
                        # print(f"TARGET:\t{target.mean()} {target.max()} {target.min()}")

                        # rmse_logits = normden.denormalize(rmse_logits)
                        # logits = normden.denormalize(logits)

                        prob, pred = torch.max(F.softmax(logits, 1), 1)
                        loss_nll   = F.nll_loss(F.log_softmax(logits, 1), states)
                        if len(quantiles)>1:
                                prob=prob.unsqueeze(1).expand_as(rmse_logits)
                                loss_mse = criterion(rmse_logits, target)
                                mae_score = F.l1_loss(rmse_logits,target.unsqueeze(1).expand_as(rmse_logits))
                        else:    
                                loss_mse = F.mse_loss(rmse_logits, target)
                                mae_score = F.l1_loss(rmse_logits, target)

                        # UNETNILM
                        loss = loss_nll + loss_mse
                        res = f1_score(pred, states, task="multiclass", num_classes=5)
                        logs = {"nlloss":loss_nll, "mseloss":loss_mse,
                                "mae":mae_score, "F1": res}
                        
                        train_logs = {}
                        for key, value in logs.items():
                                train_logs[f'tra_{key}']=value.item()

                        # # on the last batch store the stats for the epoch
                        # if train_step >= len(train_loader)-2:
                        #         if len(target.data.shape) <= 2:
                        #                 classerr.add(target.data, target)
                        #         else:
                        #                 classerr.add(target.data.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=2),
                        #                                 target.flatten())
                        #         acc_stats.append([classerr.value(1), classerr.value(min(num_classes, 5))])

                        # add the loss for each batch
                        losses["objective_loss"].add(loss.item())

                        # reset the optimizer
                        optimizer.zero_grad()

                        # backwards pass and parameter update
                        loss.backward()
                        optimizer.step()
                
                        # track batch stats
                        batch_time.add(time.time() - end)
                        steps_completed = (train_step+1)

                        # log stats every 10 batches
                        if steps_completed % print_freq == 0 or steps_completed == steps_per_epoch:
                        # Log some statistics
                                errs = OrderedDict()
                                stats_dict = OrderedDict()
                                stats_dict.update(train_logs)

                                for loss_name, meter in losses.items():
                                        stats_dict[loss_name] = meter.mean
                                stats_dict.update(errs)

                                stats_dict['LR'] = optimizer.param_groups[0]['lr']
                                stats_dict['Time'] = batch_time.mean
                                stats = ('Performance/Training/', stats_dict)
                                params = None
                                distiller.log_training_progress(stats,
                                                                params,
                                                                epoch, steps_completed,
                                                                steps_per_epoch, print_freq,
                                                                all_loggers)
                        end = time.time()

                for test_step, (inputs, target, states) in enumerate(test_loader):

                        inputs, target, states = inputs.to(device), target.to(device), states.to(device)

                        # inputs = normden.normalize(inputs)

                        # Test
                        with torch.no_grad():
                                logits, pred_power  = model(inputs)

                        # pred_power = normden.denormalize(pred_power)
                        # logits = normden.denormalize(logits)

                        prob, pred_state = torch.max(F.softmax(logits, 1), 1)
                        if len(quantiles)>1:
                                prob=prob.unsqueeze(1).expand_as(pred_power)

                        logs = {"pred_power":pred_power, "pred_state":pred_state, "power":target, "state":states}

                        outputs.append(logs)

                msglogger.info('--- test (epoch=%d)-----------', epoch)

                outputs = test_epoch_end(outputs)
                msglogger.info(str(outputs))

                # after a training epoch, do validation
                msglogger.info('--- validate (epoch=%d)-----------', epoch)

                validation_logs = validate(val_loader, model, criterion, [pylogger], epoch, tflogger)
                msglogger.info(str(validation_logs))

                perf_scores_history.append(distiller.MutableNamedTuple({'val_loss': validation_logs["log"]["val_loss"],
                                                                'epoch': epoch}))
                # # Keep perf_scores_history sorted from best to worst
                # # Sort by top1 as main sort key, then sort by top5 and epoch
                # perf_scores_history.sort(key=operator.attrgetter('top1', 'top5', 'epoch'),reverse=True)
                perf_scores_history.sort(key=operator.attrgetter('val_loss', 'epoch'))

                is_best = epoch == perf_scores_history[0].epoch

                apputils.save_checkpoint(epoch, name, model, optimizer=optimizer,
                                                scheduler=compression_scheduler, extras={},
                                                is_best=is_best, name=name,
                                                dir=msglogger.logdir)

                ms_lr_scheduler.step(metrics=validation_logs["log"]["val_loss"])
                
                # NOTE: Uncomment if using MultiStepLR scheduler
                # ms_lr_scheduler.step()

        validation_logs = validate(val_loader, model, criterion, [pylogger], epoch, tflogger)
        msglogger.info('--- validate (epoch=%d)-----------', epoch)
        msglogger.info(str(validation_logs))

        