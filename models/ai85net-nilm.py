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
            dimensions=(100, 1),  # pylint: disable=unused-argument
            bias=True,
            **kwargs
    ):
        super().__init__()

        dropout = 0.1
        hidden_layer = 256

        # self.dropout = nn.Dropout(dropout)

        self.dropout = nn.Dropout(dropout)

        self.prep0 = ai8x.FusedConv1dBNReLU(num_channels, 64, 1, stride=1, padding=0,
                                            bias=bias, batchnorm='NoAffine', **kwargs)
        self.prep1 = ai8x.FusedConv1dBNReLU(64, 64, 1, stride=1, padding=0,
                                            bias=bias, batchnorm='NoAffine', **kwargs)
        self.prep2 = ai8x.FusedConv1dBNReLU(64, 32, 1, stride=1, padding=0,
                                            bias=bias, batchnorm='NoAffine', **kwargs)

        self.enc1 = ai8x.FusedMaxPoolConv1dBNReLU(32,
                                                  16,
                                                  3, stride=1, padding=1, bias=bias, 
                                                  batchnorm='NoAffine', **kwargs)

        self.enc2 = ai8x.FusedMaxPoolConv1dBNReLU(16,
                                                  16,
                                                  3, stride=1, padding=1, bias=bias, 
                                                  batchnorm='NoAffine', **kwargs)

        self.enc3 = ai8x.FusedConv1dBNReLU(16,
                                           32,
                                           3, stride=1, padding=1, bias=bias, 
                                           batchnorm='NoAffine', **kwargs)

        self.avg = ai8x.FusedAvgPoolConv1dBNReLU(32,
                                               64,
                                               7, stride=1, padding=1, bias=bias, **kwargs)

        self.mlp1 = ai8x.Linear(512,
                                hidden_layer,
                                bias=bias, **kwargs)

        self.fc_out_state  = ai8x.Linear(hidden_layer, num_classes*2, bias=bias, wide=True, **kwargs)


    def forward(self, x):
        B = x.size(0)

        conv_out = self.prep0(x)
        conv_out = self.prep1(conv_out)
        conv_out = self.prep2(conv_out)

        conv_out = self.enc1(conv_out)
        conv_out = self.enc2(conv_out)
        conv_out = self.enc3(conv_out)
        # conv_out = self.enc4(conv_out)
        # conv_out = self.enc5(conv_out)
        # conv_out = self.enc6(conv_out)
        conv_out = self.dropout(conv_out)
        conv_out = self.avg(conv_out)
        conv_out = conv_out.view(conv_out.size(0), -1)
        conv_out = self.mlp1(conv_out)
        conv_out = self.dropout(conv_out)
        states_logits = self.fc_out_state(conv_out).reshape(B, 2, -1)

        return states_logits


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