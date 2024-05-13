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
        