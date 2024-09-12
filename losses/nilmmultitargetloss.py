
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


class NILMMultiTargetLoss(torch.nn.Module):
    
        def __init__(self, states_loss, quantiles=[0.0025,0.1, 0.5, 0.9, 0.975],
                        *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.quantiles = quantiles
                self.logsoftmax = torch.nn.LogSoftmax(dim=1)
                self.softmax = torch.nn.Softmax(dim=1)
                self.states_loss = states_loss
                self.rmse_loss = QuantileLoss()

        def forward(self, inputs, targets):
                B = inputs.size(0)

                input_state = inputs[:,:2*5].reshape(B, 2, -1)
                input_power = inputs[:,2*5:].reshape(B, 5, -1)

                target_state = targets[0]
                target_power = targets[1]

                prob, pred = torch.max(self.softmax(input_state), 1)

                ## States Loss
                loss_nll = self.states_loss(self.logsoftmax(input_state), target_state)

                ## Power Loss
                prob = prob.unsqueeze(1).expand_as(input_power)
                loss_mse = self.rmse_loss(input_power, target_power)

                loss = loss_nll + loss_mse

                return loss