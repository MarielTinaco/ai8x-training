
import os
import sys
import datetime
import time
import json

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
from tqdm import tqdm

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

import parse_qat_yaml

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'models'))
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'datasets'))

from datasets import nilm
import ai8x
mod = importlib.import_module("models.ai87net-unetnilm")

MSGLOGGER = None
PYLOGGER = None
TFLOGGER = None
ALL_LOGGERS = None

from torchmetrics.functional import f1_score

from unetnilm.metrics import get_results_summary
from unetnilm.utils import QuantileLoss

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

class NILMnet(object):

        def __init__(self, hparams):
                self.hparams = hparams
               
                with open(f"data/NILM/ukdale/{self.hparams.profile}/metadata.json") as infile:
                        metadata = json.load(infile)

                self.appliance_data = metadata["appliance_data"]
                self.appliances = metadata["appliances"]

                self.dataset_name = self.hparams.profile.split("_")[1]
                self.num_classes = len(self.appliances)
                self.model_name = self.hparams.model

                # Get policy for quantization aware training
                self.qat_policy = parse_qat_yaml.parse(hparams.qat_policy) \
                        if hparams.qat_policy.lower() != "none" else None

                self.data_path = "data/NILM/"

                self.train_loader = None
                self.val_loader = None
                self.test_loader = None

                self.setup_loggers()

                self.dataset_fn = nilm.ukdale_get_datasets

                self.setup_dataloaders()

                #################################################################################################

                self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                print('Running on device: {}'.format(self.device))

                ###################################################################################################

                self.criterion = QuantileLoss(quantiles=self.hparams.quantiles).to(self.device)

                ai8x.set_device(device=85, simulate=False, round_avg=False)

                if self.hparams.model in set(["cnn1dnilm", "ai851dnilm", None]):
                        self.model = mod.AI85CNN1DNiLM(in_size=1, 
                                  output_size=self.num_classes, 
                                  n_layers=hparams.n_layers, 
                                  d_model=hparams.d_model, 
                                  n_quantiles=len(hparams.quantiles), 
                                  dropout=hparams.dropout,
                                  pool_filter=hparams.pool_filter)

                elif self.hparams.model in set(["cnn1dnilm-states", "ai851dnilm-states"]):
                        self.model = mod.AI85CNN1DNiLMStates(in_size=1, 
                                  output_size=self.num_classes, 
                                  n_layers=hparams.n_layers, 
                                  d_model=hparams.d_model, 
                                  n_quantiles=len(hparams.quantiles), 
                                  dropout=hparams.dropout,
                                  pool_filter=hparams.pool_filter)

        def _step(self, batch):
                x, y, z = batch
                B = x.size(0)
                if self.hparams.model in set(["cnn1dnilm", "ai851dnilm", None]):
                        logits, rmse_logits = self.model(x)
                        prob, pred = torch.max(F.softmax(logits, 1), 1)
                        loss_nll   = F.nll_loss(F.log_softmax(logits, 1), z)
                        
                        if len(self.hparams.quantiles)>1:
                                prob=prob.unsqueeze(1).expand_as(rmse_logits)
                                loss_mse = self.criterion(rmse_logits, y)
                                mae_score = F.l1_loss(rmse_logits,y.unsqueeze(1).expand_as(rmse_logits))
                        else:    
                                loss_mse = F.mse_loss(rmse_logits, y)
                                mae_score = F.l1_loss(rmse_logits, y)
                        
                        loss = loss_nll + loss_mse

                        res = f1_score(pred, z, task="multiclass", num_classes=self.num_classes)
                        logs = {"nlloss":loss_nll, "mseloss":loss_mse,
                                "mae":mae_score, "F1": res}

                elif self.hparams.model in set(["cnn1dnilm-states", "ai851dnilm-states"]):
                        logits = self.model(x)
                        prob, pred = torch.max(F.softmax(logits, 1), 1)
                        loss_nll   = F.nll_loss(F.log_softmax(logits, 1), z)

                        loss = loss_nll

                        res = f1_score(pred, z, task="multiclass", num_classes=self.num_classes)
                        logs = {"nlloss" : loss_nll, "F1" : res}

                return loss, logs

        def training_step(self, batch, batch_idx):
                loss , logs = self._step(batch)
                train_logs = {}
                for key, value in logs.items():
                        train_logs[f'tra_{key}']=value.item()
                return {'loss': loss, 'log': train_logs}
        
        def validation_step(self, batch, batch_idx):
                loss, logs = self._step(batch)
                return {'loss': loss, 'log': logs}

        def test_step(self, batch,batch_idx):
                x, y, z = batch
                B = x.size(0)

                if self.hparams.model in set(["cnn1dnilm", "ai851dnilm", None]):
                        with torch.no_grad():
                                logits, pred_power  = self.model(x)
                        
                        prob, pred_state = torch.max(F.softmax(logits, 1), 1)
                        if len(self.hparams.quantiles)>1:
                                prob=prob.unsqueeze(1).expand_as(pred_power)

                        # else: 
                        logs = {"pred_power":pred_power, "pred_state":pred_state, "power":y, "state":z}

                elif self.hparams.model in set(["cnn1dnilm-states", "ai851dnilm-states"]):
                        with torch.no_grad():
                                logits  = self.model(x)
                        
                        prob, pred_state = torch.max(F.softmax(logits, 1), 1)

                        logs = {"pred_power":np.nan, "pred_state":pred_state, "power":np.nan, "state":z}

                return logs

        def validation_epoch_end(self, outputs):
                # OPTIONAL
                if self.hparams.model in set(["cnn1dnilm", "ai851dnilm", None]):
                        avg_loss = np.mean([x['nlloss'].item()+x['mseloss'].item() for x in outputs])
                        avg_f1 = np.mean([x['F1'].item() for x in outputs])
                        avg_rmse = np.mean([x['mae'].item() for x in outputs])
                        logs = {'val_loss': avg_loss, "val_F1": avg_f1, "val_mae":avg_rmse}
                elif self.hparams.model in set(["cnn1dnilm-states", "ai851dnilm-states"]):
                        avg_loss = np.mean([x['nlloss'].item() for x in outputs])
                        avg_f1 = np.mean([x['F1'].item() for x in outputs])
                        logs = {'val_loss': avg_loss, "val_F1": avg_f1}
                return {'log':logs}
        

        def test_epoch_end(self, outputs):
                appliance_data = self.appliance_data

                if self.hparams.model in set(["cnn1dnilm", "ai851dnilm", None]):
                        pred_power = torch.cat([x['pred_power'] for x in outputs], 0).cpu().numpy()
                        pred_state = torch.cat([x['pred_state'] for x in outputs], 0).cpu().numpy().astype(np.int32)
                        power = torch.cat([x['power'] for x in outputs], 0).cpu().numpy()
                        state = torch.cat([x['state'] for x in outputs], 0).cpu().numpy().astype(np.int32)
                        
                        for idx, app in enumerate(self.appliances):
                                power[:,idx] = (power[:, idx] * (appliance_data[app]["max"] - appliance_data[app]["min"])) + appliance_data[app]['min']
                                if len(self.hparams.quantiles)>=2:
                                        pred_power[:,:, idx] = (pred_power[:,:, idx] * (appliance_data[app]["max"] - appliance_data[app]["min"])) + appliance_data[app]['min']
                                        pred_power[:,:, idx] = np.where(pred_power[:,:, idx]<0, 0, pred_power[:,:, idx])
                                else:
                                        pred_power[:, idx] = (pred_power[:, idx] * (appliance_data[app]["max"] - appliance_data[app]["min"])) + appliance_data[app]['min']
                                        pred_power[:, idx] = np.where(pred_power[:, idx]<0, 0, pred_power[:, idx])    

                        if len(self.hparams.quantiles)>=2:
                                idx = len(self.hparams.quantiles)//2
                                y_pred = pred_power[:,idx]
                        else:
                                y_pred = pred_power
                        
                        per_app_results, avg_results = get_results_summary(state, pred_state, 
                                                                                power, y_pred,
                                                                                self.appliances, 
                                                                                self.dataset_name)  
                        logs = {"pred_power":np.round(y_pred, decimals=2),
                                "pred_state":pred_state, 
                                "power":np.round(power, decimals=2),
                                "state":state,
                                'app_results':per_app_results, 
                                'avg_results':avg_results}

                elif self.hparams.model in set(["cnn1dnilm-states", "ai851dnilm-states"]):
                        pred_state = torch.cat([x['pred_state'] for x in outputs], 0).cpu().numpy().astype(np.int32)
                        state = torch.cat([x['state'] for x in outputs], 0).cpu().numpy().astype(np.int32)

                        logs = {"pred_state":pred_state,
                                "state":state}
                return logs

        def fit(self):
                ####################################################################
                optims, schedulers = self.configure_optimizers()

                MSGLOGGER.info('dataset_name:%s\ndataset_fn=%s\nnum_classes=%d\nmodel_name=%s\seq_len=%s\nbatch_size=%d\nvalidation_split=%s\nlr=%f',
                                self.dataset_name,self.dataset_fn,self.num_classes,self.hparams.model,self.hparams.seq_len,self.hparams.batch_size, self.hparams.validation_split,self.hparams.lr)

                MSGLOGGER.info('model: %s',self.model)
                self.model = self.model.to(self.device)

                # start the clock
                tic = datetime.datetime.now()

                perf_scores_history = []

                # training loop
                for epoch in range(self.hparams.epochs):

                        if epoch > 0 and epoch == self.qat_policy['start_epoch']:
                                print('QAT is starting!')
                                # Fuse the BN parameters into conv layers before Quantization 
                                ai8x.fuse_bn_layers(self.model)
                        
                                # Switch model from unquantized to quantized for QAT
                                ai8x.initiate_qat(self.model, self.qat_policy)

                                # self.Model is re-transferred to GPU in case parameters were added
                                self.model.to(self.device)

                                # Empty the performance scores list for QAT operation
                                perf_scores_history = []
                                self.model_name = f'{self.model_name}_qat'
                        
                        # store loss and training stats
                        losses = {'objective_loss': tnt.AverageValueMeter()}
                        classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, min(self.num_classes, 5)))
                        batch_time = tnt.AverageValueMeter()
                        data_time = tnt.AverageValueMeter()

                        # logging stats
                        total_samples = len(self.train_loader.sampler)
                        batch_size = self.train_loader.batch_size
                        steps_per_epoch = (total_samples + batch_size - 1) // batch_size
                        MSGLOGGER.info('Training epoch: %d samples (%d per mini-batch)', total_samples, batch_size)

                        self.model.train()

                        end = time.time()

                        for train_step, train_batch in enumerate(self.train_loader):
                                torch.cuda.empty_cache()
                                train_batch = train_batch[0].to(self.device), train_batch[1].to(self.device), train_batch[2].to(self.device)
                                
                                logs = self.training_step(train_batch, train_step)
                                
                                loss = logs['loss']
                                train_logs = logs['log']

                                losses["objective_loss"].add(loss.item())

                                # reset the optimizer
                                optims[0].zero_grad()

                                # backwards pass and parameter update
                                loss.backward()
                                optims[0].step()
                        
                                # track batch stats
                                batch_time.add(time.time() - end)
                                steps_completed = (train_step+1)

                                # log stats every 10 batches
                                if steps_completed % self.hparams.print_freq == 0 or steps_completed == steps_per_epoch:
                                # Log some statistics
                                        errs = OrderedDict()
                                        stats_dict = OrderedDict()
                                        stats_dict.update(train_logs)

                                        for loss_name, meter in losses.items():
                                                stats_dict[loss_name] = meter.mean
                                        stats_dict.update(errs)

                                        stats_dict['LR'] = optims[0].param_groups[0]['lr']
                                        stats_dict['Time'] = batch_time.mean
                                        stats = ('Performance/Training/', stats_dict)
                                        params = None
                                        distiller.log_training_progress(stats,
                                                                        params,
                                                                        epoch, steps_completed,
                                                                        steps_per_epoch, self.hparams.print_freq,
                                                                        ALL_LOGGERS)

                                end = time.time()

                        test_outputs = []
                        MSGLOGGER.info('--- test (epoch=%d)-----------', epoch)

                        for testing_step, test_batch in enumerate(self.test_loader):
                                test_batch = test_batch[0].to(self.device), test_batch[1].to(self.device), test_batch[2].to(self.device)
                                logs = self.test_step(test_batch, testing_step)
                                test_outputs.append(logs)

                        test_logs = self.test_epoch_end(test_outputs)
                        MSGLOGGER.info(str(test_logs))

                        # after a training epoch, do validation
                        MSGLOGGER.info('--- validate (epoch=%d)-----------', epoch)
                        val_logs = self.validate(epoch)

                        MSGLOGGER.info(str(val_logs))

                        perf_scores_history.append(distiller.MutableNamedTuple({'val_loss': val_logs['log']["val_loss"],
                                                                'epoch': epoch}))

                        # # Keep perf_scores_history sorted from best to worst
                        # # Sort by top1 as main sort key, then sort by top5 and epoch
                        # perf_scores_history.sort(key=operator.attrgetter('top1', 'top5', 'epoch'),reverse=True)
                        perf_scores_history.sort(key=operator.attrgetter('val_loss', 'epoch'))

                        is_best = epoch == perf_scores_history[0].epoch

                        apputils.save_checkpoint(epoch, self.model_name, self.model, optimizer=optims[0],
                                                        scheduler=schedulers[1], extras={},
                                                        is_best=is_best, name=self.model_name,
                                                        dir=MSGLOGGER.logdir)

                        schedulers[0].step(metrics=val_logs['log']["val_loss"])
                        schedulers[1].on_epoch_end(epoch, optims[0])

                pred = self.predict(self.model, self.val_loader)
                MSGLOGGER.info(str(pred))

        def validate(self, epoch=-1):
                """Execute the validation/test loop."""

                # store loss stats
                losses = {'objective_loss': tnt.AverageValueMeter()}
                classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, min(self.num_classes, 5)))

                # validation set info
                batch_time = tnt.AverageValueMeter()
                total_samples = len(self.val_loader.sampler)
                batch_size = self.val_loader.batch_size
                confusion = tnt.ConfusionMeter(self.num_classes)
                total_steps = (total_samples + batch_size - 1) // batch_size
                steps_per_epoch = (total_samples + batch_size - 1) // batch_size
                
                MSGLOGGER.info('%d samples (%d per mini-batch)', total_samples, batch_size)

                end = time.time()

                outputs = []

                # Switch to evaluation mode
                self.model.eval()

                for validation_step, validation_batch in enumerate(self.val_loader):
                        with torch.no_grad():
                                validation_batch = tuple(data.to(self.device) for data in validation_batch)
                                logs = self.validation_step(validation_batch, validation_step)
                                loss = logs['loss']

                                losses['objective_loss'].add(loss.item())

                                batch_time.add(time.time() - end)
                                steps_completed = (validation_step+1)
                                if steps_completed % self.hparams.print_freq == 0 or steps_completed == total_steps:
                                        stats = ('Performance/Validation/',
                                                OrderedDict([('Loss', losses['objective_loss'].mean),('Time', batch_time.mean)]))
                                        params = None
                                        distiller.log_training_progress(stats,
                                                                        params,
                                                                        epoch, steps_completed,
                                                                        steps_per_epoch, self.hparams.print_freq,
                                                                        ALL_LOGGERS)

                                # MSGLOGGER.info(str(logs['log']))

                                outputs.append(logs['log'])

                                end = time.time()

                return self.validation_epoch_end(outputs)

        def predict(self, model, dataloader : DataLoader):
                outputs = []

                model = model.eval()
                batch_size   = dataloader.batchsize if hasattr(dataloader, 'len') else dataloader.batch_size
                num_batches = len(dataloader.sampler)
                values = range(num_batches)

                with tqdm(total=len(values), file=sys.stdout) as pbar:
                        with torch.no_grad():
                                for batch_idx, batch in enumerate(dataloader):
                                        batch = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                                        logs = self.test_step(batch, batch_idx)
                                        outputs.append(logs)
                                        del  batch
                                        pbar.set_description('processed: %d' % (1 + batch_idx))
                                        pbar.update(batch_size)
                                pbar.close()

                return self.test_epoch_end(outputs)                

        def setup_loggers(self):
                global MSGLOGGER, PYLOGGER, TFLOGGER, ALL_LOGGERS
                ####################################################################################

                log_dir = self.hparams.log_dir
                log_prefix = f"{self.dataset_name}-train"

                MSGLOGGER = apputils.config_pylogger('logging.conf', log_prefix,
                                                        log_dir)

                PYLOGGER = PythonLogger(MSGLOGGER, log_1d=True)
                ALL_LOGGERS = [PYLOGGER]

                # tensorboard
                TFLOGGER = TensorBoardLogger(MSGLOGGER.logdir, log_1d=True, comment='_'+self.dataset_name)

                TFLOGGER.tblogger.writer.add_text('Command line', "args ---")

                ALL_LOGGERS.append(TFLOGGER)
                all_tbloggers = [TFLOGGER]

                ##########################################################################################

        def setup_dataloaders(self):
                ############################################################################################
                self.args = Args(act_mode_8bit=False)

                self.train_loader, self.val_loader, self.test_loader, _ = apputils.get_data_loaders(
                        self.dataset_fn, (self.data_path,self.args), self.hparams.batch_size,
                        self.hparams.workers, self.hparams.validation_split, self.hparams.deterministic,1, 1, 1)
                MSGLOGGER.info('Dataset sizes:\n\ttraining=%d\n\tvalidation=%d\n\ttest=%d',
                                len(self.train_loader.sampler), len(self.val_loader.sampler), len(self.test_loader.sampler))
                MSGLOGGER.info('Augmentations:%s',self.train_loader.dataset.transform)


        def configure_optimizers(self):
                ########################################################################################################

                MSGLOGGER.info('epochs: %d',self.hparams.epochs)
                optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta_1, self.hparams.beta_2))
                MSGLOGGER.info('Optimizer Type: %s', type(optimizer))
                # ms_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 35,100], gamma=0.5)
                # MSGLOGGER.info("lr_schedule:%s","base: "+str(ms_lr_scheduler.base_lrs)+" milestones: "+str(ms_lr_scheduler.milestones)+ " gamma: "+str(ms_lr_scheduler.gamma))
                ms_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.hparams.scheduler, verbose=True, min_lr=1e-6, mode="min")

                compression_scheduler = distiller.CompressionScheduler(self.model)

                return [optimizer], [ms_lr_scheduler, compression_scheduler]



if __name__ == "__main__":
        import argparse

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--lr', default=1.e-4, type=float)
        parser.add_argument('--batch_size', default=256, type=int)
        parser.add_argument('--beta_1', default=0.999, type=float)
        parser.add_argument('--beta_2', default=0.98, type=float)
        parser.add_argument('--quantiles', default=[0.0025,0.1, 0.5, 0.9, 0.975], type=list)
        parser.add_argument('--dropout', default=0.25, type=float)
        parser.add_argument('--profile', default=sorted(os.listdir('data/NILM/'))[-1], type=str)
        parser.add_argument('--seq_len', default=100, type=int)
        parser.add_argument('--n_layers', default=5, type=int)
        parser.add_argument('--d_model', default=128, type=int)
        parser.add_argument('--pool_filter', default=8, type=int)
        parser.add_argument('--act_mode_8bit', default=False, type=bool)
        parser.add_argument('--qat-policy', default='policies/qat_policy_cnn1dnilm.yaml', type=str)
        parser.add_argument('--log_dir', default='logs', type=str)
        parser.add_argument('--model', default='cnn1dnilm', type=str)
        parser.add_argument('--validation_split', default=0.1, type=float)
        parser.add_argument('--deterministic', default=True, type=bool)
        parser.add_argument('--epochs', default=50, type=int)
        parser.add_argument('--workers', default=5, type=int)
        parser.add_argument('--scheduler', default=5, type=int)
        parser.add_argument('--print_freq', default=10, type=int)

        net = NILMnet(parser.parse_args())

        net.fit()

