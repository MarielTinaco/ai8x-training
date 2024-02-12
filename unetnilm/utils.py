import glob
import os
import torch

class QuantileLoss(torch.nn.Module):
    def __init__(self, quantiles=[0.0025,0.1, 0.5, 0.9, 0.975]):
        self.quantiles = quantiles
        super().__init__() 
    def forward(self, inputs, targets):
        targets = targets.unsqueeze(1).expand_as(inputs)
        quantiles = torch.tensor(self.quantiles).float().to(targets.device)
        error = (targets - inputs).permute(0,2,1)
        loss = torch.max(quantiles*error, (quantiles-1)*error)
        return loss.mean()
        

class ObjectDict(dict):
    """
    Interface similar to an argparser
    """
    def __init__(self):
        pass
    
    def __setattr__(self, attr, value):
        self[attr] = value
        return self[attr]
    
    def __getattr__(self, attr):
        if attr.startswith('_'):
            # https://stackoverflow.com/questions/10364332/how-to-pickle-python-object-derived-from-dict
            raise AttributeError
        return dict(self)[attr]
    
    @property
    def __dict__(self):
        return dict(self)

def get_latest_checkpoint(checkpoint_path):
    checkpoint_path = str(checkpoint_path)
    list_of_files = glob.glob(checkpoint_path + '/*.ckpt')
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
    else:
        latest_file = None
    return latest_file
