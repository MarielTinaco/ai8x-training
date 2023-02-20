import os
from pathlib import Path

#raw_data_path = Path("C:/Users/J_C/Desktop/DATASETS/raw")
raw_data_path = "C:/Users/J_C/Desktop/DATASETS/raw/"
class_file_count = {}


#class_dirs = [d for d in raw_data_path.iterdir() if d.is_dir() and (d.stem != "_background_noise_")  and (d.stem != "horse_cough")]
class_dirs = []
for d in os.listdir(raw_data_path):
    if os.path.isdir(raw_data_path+d):
        class_dirs.append(raw_data_path+d+'/')

# Create combined Dataset
import shutil
combined_path = "C:/Users/J_C/Documents/GitHub/ai8x-training/data/KWS_EQUINE/raw/combined/"
if os.path.isdir(combined_path) == False:
    os.makedirs(combined_path)

for i,d in enumerate(class_dirs):
    fnames =  os.listdir(d)
    for fi,f in enumerate(os.listdir(d)):
        print(d,'---',f)
        shutil.copyfile(d+f, combined_path+'/'+str(i)+'_'+fnames[fi])