
import torch
import torch.nn.functional as F
from torch import nn

class MultiTargetNllLoss(nn.Module):
    
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = device

    def forward(self, output, target):
        loss_nll   = F.nll_loss(F.log_softmax(output, 1), target)
        return loss_nll
