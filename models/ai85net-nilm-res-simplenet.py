###################################################################################################
#
# Copyright (C) 2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Seq2Point network for MAX7800X
"""
import torch
from torch import nn

import torch.nn.functional as F

import ai8x


class AI85NILMResSimpleNet(nn.Module):
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

                dropout = 0.2

                ## INIT ##
                self.prep = ai8x.FusedConv1dBNReLU(num_channels, 16, 3, stride=1, padding=1,
                        bias=bias, batchnorm='Affine', **kwargs)

                ## 1ST ##
                self.conv1_skip = ai8x.FusedMaxPoolConv1dBNReLU(16, 24, 1, stride=1, padding=1,
                        pool_stride=1, pool_size=3, bias=bias, batchnorm='Affine', **kwargs)
                self.conv1_1 = ai8x.FusedConv1dBNReLU(16, 24, 3, stride=1, padding=1,
                        bias=bias, batchnorm='Affine', **kwargs)
                self.dropout1_1 = nn.Dropout(dropout)
                self.conv1_2 = ai8x.FusedConv1dBNReLU(24, 24, 3, stride=1, padding=1,
                        bias=bias, batchnorm='Affine', **kwargs)
                self.dropout1_2 = nn.Dropout(dropout)
                self.resid_1 = ai8x.Add()

                ## 2ND ##
                self.conv2_skip = ai8x.FusedMaxPoolConv1dBNReLU(24, 32, 3, stride=1, padding=0,
                        pool_stride=1, pool_size=3, bias=bias, batchnorm='Affine', **kwargs)
                self.conv2_1 = ai8x.FusedConv1dBNReLU(24, 32, 3, stride=1, padding=0,
                        bias=bias, batchnorm='Affine', **kwargs)
                self.dropout2_1 = nn.Dropout(dropout)
                self.conv2_2 = ai8x.FusedConv1dBNReLU(32, 32, 3, stride=1, padding=0,
                        bias=bias, batchnorm='Affine', **kwargs)
                self.dropout2_2 = nn.Dropout(dropout)
                self.resid_2 = ai8x.Add()

                last_layer = 32

                self.conv10 = ai8x.FusedMaxPoolConv1dBNReLU(last_layer, int(last_layer*0.8), 3, pool_stride=4,
                                                            padding=0, bias=bias, batchnorm='Affine', **kwargs)

                self.lin = ai8x.FusedLinearReLU(550, 128, bias=bias, **kwargs)

                ## MLP ##
                self.mlp1 = ai8x.FusedLinearReLU(128, 256, bias=bias, **kwargs)

                ## LINEAR ##
                self.fc_state = ai8x.Linear(256, num_classes*2, bias=bias, **kwargs)
                self.fc_power = ai8x.Linear(256, num_classes*5, bias=bias, **kwargs)

                self.initWeights("kaiming")

        def forward(self, x):
                                                        # 100
                x = self.prep(x)                        # 16x100

                # resblock 1
                x_res = self.conv1_1(x)                 # 24x100
                x_res = self.dropout1_1(x_res)          # 24x100
                x_res = self.conv1_2(x_res)             # 24x100
                x_res = self.dropout1_2(x_res)          # 24x100
                x = self.conv1_skip(x)                  # 24x100
                x = self.resid_1(x, x_res)              # 24x100

                # resblock 2
                x_res = self.conv2_1(x)                 # 32x98
                x_res = self.dropout2_1(x_res)          # 32x98
                x_res = self.conv2_2(x_res)             # 32x98
                x_res = self.dropout2_2(x_res)          # 32x98
                x = self.conv2_skip(x)                  # 32x98
                x = self.resid_2(x, x_res)              # 32x96

                x = self.conv10(x)                      # (last_layer*0.8)x(48)

                x = x.view(x.size(0), -1)               # 768
                x = self.lin(x)                         # 128
                x = self.mlp1(x)                        # 256

                x1 = self.fc_state(x)           
                x1 = x1.view(x1.size(0), -1)            # 10
                x2 = self.fc_power(x)           
                x2 = x2.view(x2.size(0), -1)            # 25
                x = torch.cat([x1, x2], dim=1)          # 35
                return x

        def initWeights(self, weight_init="kaiming"):
                """
                Auto Encoder Weight Initialization
                """
                weight_init = weight_init.lower()
                assert weight_init in ('kaiming', 'xavier', 'glorot')

                for m in self.modules():
                        if isinstance(m, nn.Conv2d):
                                if weight_init == "kaiming":
                                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                                elif weight_init in ('glorot', 'xavier'):
                                        nn.init.xavier_uniform_(m.weight)

                        elif isinstance(m, nn.ConvTranspose2d):
                                if weight_init == "kaiming":
                                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                                elif weight_init in ('glorot', 'xavier'):
                                        nn.init.xavier_uniform_(m.weight)

                        elif isinstance(m, nn.Linear):
                                if weight_init == "kaiming":
                                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                                elif weight_init in ('glorot', 'xavier'):
                                        nn.init.xavier_uniform_(m.weight)

def ai85nilmressimplenet(pretrained=False, **kwargs):
        """
        Constructs a AI85KWS20Net model.
        """
        assert not pretrained
        return AI85NILMResSimpleNet(**kwargs)


models = [
        {
                'name': 'ai85nilmressimplenet',
                'min_input': 1,
                'dim': 1,
        },
]