import importlib
import sys
import os
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader

from torch.utils import data
from torchvision import transforms

sys.path.append(os.path.join(os.getcwd(), ".."))

from distiller import apputils
import ai8x

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

kws20 = importlib.import_module("datasets.kws20-horsecough")

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support, roc_auc_score, roc_curve


import librosa
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# FIX SEED FOR REPRODUCIBILITY
seed = 69
torch.manual_seed(seed)

def rescale(audio, min_val=-1,max_val=1):
    sig = audio
    mean = np.average(sig)

    sig = sig-mean # REMOVE DC COMPONENT

    sig_max = np.max(sig)
    sig_min = np.min(sig)

    if sig_max >= np.abs(sig_min):
        sig_scaled = sig/sig_max
    else:
        sig_scaled = sig/np.abs(sig_min)

    return sig_scaled

def rescale2(audio, min_val=-1,max_val=1):
    scaler = MinMaxScaler(feature_range=(min_val,max_val))
    audio = audio.reshape(-1,1)
    scaler.fit(audio)
    scaled = np.array(scaler.transform(audio))
    
    return scaled[:,0]


def plot_confusion(y_true, y_pred, classes):
    cf_matrix = confusion_matrix(y_true = y_true, y_pred = y_pred, labels =list(range(len(classes))))
    print(cf_matrix)

def plot_roc_curve(true_y, y_prob):
    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

def custom_dataloader(data, train_idx, val_idx, test_idx, batch_size = 2048):
    data_file = data
    x = np.asarray(data_file[0][test_idx])
    y = np.squeeze(np.asarray(data_file[1][test_idx]))
    test_set = TensorDataset(torch.Tensor(x),torch.Tensor(y).to(dtype =torch.int64))
    test_loader = DataLoader(test_set,batch_size=batch_size, num_workers=4, pin_memory=True)

    x = np.asarray(data_file[0][val_idx])
    y = np.squeeze(np.asarray(data_file[1][val_idx]))
    val_set = TensorDataset(torch.Tensor(x),torch.Tensor(y).to(dtype =torch.int64))
    val_loader = DataLoader(val_set,batch_size=batch_size, num_workers=4, pin_memory=True)

    x = np.asarray(data_file[0][train_idx])
    y = np.squeeze(np.asarray(data_file[1][train_idx]))
    train_set = TensorDataset(torch.Tensor(x),torch.Tensor(y).to(dtype =torch.int64))
    train_loader = DataLoader(train_set,batch_size=batch_size, num_workers=4, pin_memory=True)

    return train_loader,val_loader,test_loader

def freeze_layer(layer):
    for p in layer.parameters():
        p.requires_grad = False

def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    return params

class sys():
    def __init__(self,rx_buffer):
        self.SAMPLE_SIZE = 16384
        self.SILENCE_COUNTER_THRESHOLD = 20
        self.THRESHOLD_HIGH = 130 
        self.THRESHOLD_LOW = 70
        self.rx = rx_buffer
        

    def silence_detect(self,counter): #:566
        threshold_low = self.THRESHOLD_LOW
        SAMPLE_SIZE = self.SAMPLE_SIZE
        ai85Counter = counter
            
        if avg < threshold_low and ai85Counter >= SAMPLE_SIZE / 3:
            avgSilenceCounter += 1
        else:
            avgSilenceCounter = 0


        pass

    def start_cnn(): #:599
        pass


        
    
if __name__ == "__main__":
    CHUNK = 128 
    PREAMBLE_SIZE = int(30 * CHUNK)


    ### CNN INIT ###
    if torch.cuda.is_available(): device = torch.device('cuda:0')
    else: device = torch.device('cpu')
        
    ai8x.set_device(device=85, simulate=False, round_avg=False)
    mod = importlib.import_module("models.ai85net-equine")
    model = mod.AI85EQUINE()

    model = model.to(device)

    print('Running on device: {}'.format(torch.cuda.get_device_name()))
    print(f'Number of Model Params: {count_params(model)}')
    print(model)

    ### MIC PREPROCESSING ###

