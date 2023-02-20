import importlib
import sys
import os
import numpy as np
import time
import shutil 

from torch import nn
import torch.optim as optim

from torch.utils import data
from torchvision import transforms

sys.path.append(os.path.join(os.getcwd(), ".."))

from pathlib import Path

raw_data_path = Path("C:/Users/J_C\Desktop/DATASETS/raw")
combined_path = "C:/Users/J_C/Desktop/DATASETS/raw/_combined"
class_file_count = {}

class_dirs = [d for d in raw_data_path.iterdir() if d.is_dir() and (d.stem != "_background_noise_" and d.stem != "_combined")]

if os.path.isdir(combined_path) == False:
    os.makedirs(combined_path)

for i,d in enumerate(class_dirs):
    if i!=13: #Skip horse cough
        class_file_count[d] = len(list(d.iterdir()))
        print(class_file_count[d] ," files in folder ",d)
        fnames =  os.listdir(d)
        for fi,f in enumerate(d.iterdir()):
            shutil.copyfile(f, combined_path+'/'+str(i)+'_'+fnames[fi])


        

