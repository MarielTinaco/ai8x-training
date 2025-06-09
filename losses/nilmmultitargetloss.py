
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
                     num_classes=5, logsoftmax_scale_factor=5,
                        *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.quantiles = quantiles
                self.logsoftmax = torch.nn.LogSoftmax(dim=1)
                self.softmax = torch.nn.Softmax(dim=1)
                self.states_loss = states_loss
                self.rmse_loss = QuantileLoss()
                self.num_classes = num_classes
                self.logsoftmax_scale_factor = logsoftmax_scale_factor

        def forward(self, inputs, targets):
                B = inputs.size(0)

                # inputs = torch.clip(inputs, min=-1)
                # inputs = (inputs + 1)/2

                input_state = inputs[:,:2*self.num_classes].reshape(B, 2, -1)
                input_power = inputs[:,2*self.num_classes:].reshape(B, len(self.quantiles), -1)

                target_state = targets[0]
                target_power = targets[1]

                ## States Loss
                # Scaling the values of the output state to stay stabilize log softmax function
                ls = self.logsoftmax(input_state * self.logsoftmax_scale_factor)
                if self.states_loss.weight:
                        loss_nll = self.states_loss(ls, target_state, weight=self.states_loss.weight)
                else:
                        loss_nll = self.states_loss(ls, target_state)

                ## Power Loss
                # prob = prob.unsqueeze(1).expand_as(input_power)
                loss_mse = self.rmse_loss(input_power, target_power)

                loss = loss_nll + loss_mse

                return loss