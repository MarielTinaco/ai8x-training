###################################################################################################
#
# Copyright (C) 2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
UNet network for MAX7800X
"""
import torch
from torch import nn

import torch.nn.functional as F

import ai8x


class AI85NILMNet(nn.Module):
    """
    Small size UNet model
    """
    def __init__(
            self,
            num_classes=5,
            num_channels=1,
            dimensions=(100,),  # pylint: disable=unused-argument
            bias=True,
            **kwargs
    ):
        super().__init__()

        d_model = 18
        n_layers = 3
        pool_filter = 4
        dropout = 0.1
        hidden_layer = 256

        self.pool_filter = pool_filter

        # self.enc_net = Encoder(n_channels=num_channels, n_kernels=d_model, n_layers=n_layers, seq_size=dimensions[0], **kwargs)
        # self.mlp_layer = MLPLayer(in_size=d_model*pool_filter, hidden_arch=[1024], output_size=None, **kwargs)
        # self.dropout = nn.Dropout(dropout)
        # self.fc_out_state  = ai8x.Linear(1024, num_classes*2, bias=bias, **kwargs)

        self.dropout = nn.Dropout(dropout)

        self.enc1 = ai8x.FusedMaxPoolConv1dBNReLU(num_channels,
                                                  d_model // 2**(n_layers-1),
                                                  3, stride=1, padding=1, bias=bias, 
                                                  batchnorm='NoAffine', **kwargs)

        self.enc2 = ai8x.FusedMaxPoolConv1dBNReLU(d_model // 2**(n_layers-1),
                                                  d_model // 2**(n_layers-2),
                                                  3, stride=1, padding=1, bias=bias, 
                                                  batchnorm='NoAffine', **kwargs)

        self.enc3 = ai8x.FusedConv1dBNReLU(d_model // 2**(n_layers-2),
                                           d_model,
                                           3, stride=1, padding=1, bias=bias, 
                                           batchnorm='NoAffine', **kwargs)

        self.avg = ai8x.FusedAvgPoolConv1dReLU(d_model,
                                               d_model * pool_filter,
                                               1, stride=1, padding=1, bias=bias, **kwargs)

        self.mlp1 = ai8x.Linear(d_model * pool_filter * 14,
                                hidden_layer,
                                bias=bias, **kwargs)

        self.fc_out_state  = ai8x.Linear(hidden_layer, num_classes*2, bias=bias, wide=True, **kwargs)

        self.m = nn.LogSoftmax(dim=1)
        

    def forward(self, x):
        x = x.permute(0, 2, 1)
        B = x.size(0)
        # conv_out = self.dropout(self.enc_net(x))
        # conv_out = F.adaptive_avg_pool1d(conv_out, self.pool_filter).reshape(B, -1)
        # mlp_out  = self.dropout(self.mlp_layer(conv_out))
        # states_logits   = self.fc_out_state(mlp_out).reshape(B, 2, -1)

        conv_out = self.enc1(x)
        conv_out = self.enc2(conv_out)
        conv_out = self.enc3(conv_out)
        drop = self.dropout(conv_out)

        drop = self.avg(conv_out)
        drop = drop.view(drop.size(0), -1)

        mlp_out = self.mlp1(drop)
        drop = self.dropout(mlp_out)

        states_logits   = self.fc_out_state(drop).reshape(B, 2, -1)

        return self.m(states_logits)

class Encoder(nn.Module):
    def __init__(self, 
                 n_channels=10, 
                 n_kernels=16, 
                 n_layers=3, 
                 seq_size=50,
                 device="cuda:0",
                 **kwargs):
        super(Encoder, self).__init__()
        self.feat_size = (seq_size-1) // 2**n_layers +1
        self.feat_dim = self.feat_size * n_kernels
        self.conv_stack = nn.Sequential(
            *([Conv1D(n_channels, n_kernels // 2**(n_layers-1), activation="ReLU", pooling="Max", last=False, device=device, **kwargs)] +
              [Conv1D(n_kernels//2**(n_layers-l),
                         n_kernels//2**(n_layers-l-1), activation="ReLU", pooling="Max", last=False, device=device, **kwargs)
               for l in range(1, n_layers-1)] +
              [Conv1D(n_kernels // 2, n_kernels, activation="ReLU", pooling="Max", last=True, device=device, **kwargs)])
        )
    def forward(self, x):
        assert len(x.size())==3
        feats = self.conv_stack(x)
        return feats

class MLPLayer(nn.Module):
    def __init__(self, in_size, 
                 hidden_arch=[128/2, 512/2, 1024/2], 
                 output_size=None,
                 **kwargs):
        
        super(MLPLayer, self).__init__()
        self.in_size = in_size
        self.output_size = output_size
        layer_sizes = [in_size] + [x for x in hidden_arch]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = ai8x.Linear(layer_sizes[i], layer_sizes[i+1], **kwargs)
            self.layers.append(layer)

        if output_size is not None:
            layer = ai8x.Linear(layer_sizes[-1], output_size, **kwargs)
            self.layers.append(layer)

        self.init_weights()
        self.mlp_network =  nn.Sequential(*self.layers)

    def forward(self, z):
        return self.mlp_network(z)
        
    def init_weights(self):
        for layer in self.layers:
            try:
                if isinstance(layer, ai8x.Linear):
                    nn.utils.weight_norm(layer)
                    nn.init.xavier_uniform_(layer.weight)
            except: pass


class Conv1D(nn.Module):
    
    def __init__(self,
                 num_channels,
                 num_kernels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 pooling="Max",
                 activation="ReLU",
                 batchnorm="NoAffine",
                 last=False,
                 device="cuda:0",
                 **kwargs):
        super(Conv1D, self).__init__()
        
        if not last:
            if pooling == "Max":
                if activation == "ReLU":
                    self.net = ai8x.FusedMaxPoolConv1dBNReLU(in_channels=num_channels,
                                                             out_channels=num_kernels,
                                                             kernel_size=kernel_size,
                                                             stride=stride,
                                                             padding=padding,
                                                             bias=True,
                                                             batchnorm="NoAffine",
                                                             **kwargs)
                elif activation == "Abs":
                    self.net = ai8x.FusedMaxPoolConv1dBNAbs(in_channels=num_channels,
                                                            out_channels=num_kernels,
                                                            kernel_size=kernel_size,
                                                            stride=stride,
                                                            padding=padding,
                                                            bias=True,
                                                            batchnorm="NoAffine",
                                                            **kwargs)
                else:
                    self.net = ai8x.FusedMaxPoolConv1d(in_channels=num_channels,
                                                       out_channels=num_kernels,
                                                       kernel_size=kernel_size,
                                                       stride=stride,
                                                       padding=padding,
                                                       batchnorm=batchnorm,
                                                       **kwargs)
            elif pooling == "Avg":
                if activation == "ReLU":
                    self.net = ai8x.FusedAvgPoolConv1dBNReLU(in_channels=num_channels,
                                                             out_channels=num_kernels,
                                                             kernel_size=kernel_size,
                                                             stride=stride,
                                                             padding=padding,
                                                             bias=True,
                                                             batchnorm="NoAffine",
                                                             **kwargs)
                elif activation == "Abs":
                    self.net = ai8x.FusedAvgPoolConv1dBNAbs(in_channels=num_channels,
                                                            out_channels=num_kernels,
                                                            kernel_size=kernel_size,
                                                            stride=stride,
                                                            padding=padding,
                                                            bias=True,
                                                            batchnorm="NoAffine",
                                                            **kwargs)
                else:
                    self.net = ai8x.FusedAvgPoolConv1d(in_channels=num_channels,
                                                       out_channels=num_kernels,
                                                       kernel_size=kernel_size,
                                                       stride=stride,
                                                       padding=padding,
                                                       batchnorm=batchnorm,
                                                    **kwargs)
            else:
                if activation == "ReLU":
                    self.net = ai8x.FusedConv1dBNReLU(in_channels=num_channels,
                                                      out_channels=num_kernels,
                                                      kernel_size=kernel_size,
                                                      stride=stride,
                                                      padding=padding,
                                                      bias=True,
                                                      batchnorm="NoAffine",
                                                      **kwargs)
                elif activation == "Abs":
                    self.net = ai8x.FusedConv1dBNAbs(in_channels=num_channels,
                                                    out_channels=num_kernels,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=padding,
                                                    bias=True,
                                                    batchnorm="NoAffine",
                                                    **kwargs)
                else:
                    self.net = ai8x.FusedAvgPoolConv1d(in_channels=num_channels,
                                                       out_channels=num_kernels,
                                                       kernel_size=kernel_size,
                                                       stride=stride,
                                                       padding=padding,
                                                       batchnorm=batchnorm,
                                                       **kwargs
                    )
        else:
            if pooling == "Max":
                self.net = ai8x.FusedMaxPoolConv1d(in_channels=num_channels,
                                                   out_channels=num_kernels,
                                                   kernel_size=kernel_size,
                                                   stride=stride,
                                                   padding=padding,
                                                   batchnorm=batchnorm,
                                                   **kwargs
                )
            elif pooling == "Avg":
                self.net = ai8x.FusedAvgPoolConv1d(
                    in_channels=num_channels,
                    out_channels=num_kernels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    batchnorm=batchnorm,
                    **kwargs
                )
            else:
                self.net = ai8x.Conv1d(
                    in_channels=num_channels,
                    out_channels=num_kernels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    batchnorm=batchnorm,
                    **kwargs
                )

        nn.utils.weight_norm(self.net.op.to(device))    
        nn.init.xavier_uniform_(self.net.op.weight)
        
    def forward(self, x):
        return self.net(x)


def ai85nilmnet(pretrained=False, **kwargs):
    """
    Constructs a AI85KWS20Net model.
    """
    assert not pretrained
    return AI85NILMNet(**kwargs)


models = [
    {
        'name': 'ai85nilmnet',
        'min_input': 1,
        'dim': 1,
    },
]