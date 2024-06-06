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
                self.prep = ai8x.FusedConv1dBNReLU(num_channels, 32, 3, stride=1, padding=2,
                        bias=bias, batchnorm='Affine', **kwargs)

                ## 1ST ##
                # Captures initial causal features but keeps the prepped data in-tact in residue
                self.conv0_1 = ai8x.FusedConv1dBNReLU(32, 32, 3, stride=1, padding=1,
                        bias=bias, batchnorm='Affine', **kwargs)

                self.conv0_2 = ai8x.FusedConv1dBNReLU(32, 32, 3, stride=1, padding=1,
                        bias=bias, batchnorm='Affine', **kwargs)

                self.conv0_3 = ai8x.FusedConv1dBNReLU(32, 32, 3, stride=1, padding=1,
                        bias=bias, batchnorm='Affine', **kwargs)
                self.resid_0 = ai8x.Add()

                ## BOTTLENECK ##
                self.downsample1 = ai8x.FusedMaxPoolConv1dBN(32, 96, 3, stride=1, padding=0,
                        pool_stride=2, pool_size=2,
                        bias=bias, batchnorm='Affine', **kwargs)
                self.conv1_1 = ai8x.FusedConv1dBNReLU(32, 48, 1, stride=1, padding=0,
                        bias=bias, batchnorm='Affine', **kwargs)
                self.conv1_2 = ai8x.FusedMaxPoolConv1dBNReLU(48, 64, 3, stride=1, padding=0,
                        bias=bias, batchnorm='Affine', **kwargs)
                self.conv1_3 = ai8x.FusedMaxPoolConv1dBN(64, 96, 1, stride=1, padding=0,
                        pool_stride=1, pool_size=1, bias=bias, batchnorm='Affine', **kwargs)
                self.dropout1_1 = nn.Dropout(dropout)
                self.dropout1_2 = nn.Dropout(dropout)
                self.resid_1 = ai8x.Add()

                ## 2ND ##
                self.conv2_1 = ai8x.FusedConv1dBNReLU(96, 128, 1, stride=1, padding=0,
                        bias=bias, batchnorm='Affine', **kwargs)
                # residual max pool 2
                self.conv2_2 = ai8x.FusedMaxPoolConv1dBNReLU(128, 128, 3, stride=1, padding=0,
                        bias=bias, batchnorm='Affine', **kwargs)
                self.conv2_3 = ai8x.FusedConv1dBNReLU(128, 128, 1, stride=1, padding=0,
                        bias=bias, batchnorm='Affine', **kwargs)
                self.resid_2 = ai8x.Add()

                self.conv_out1 = ai8x.FusedConv1dBNReLU(128, 96, 3, stride=1, padding=0,
                        bias=bias, batchnorm='Affine', **kwargs)
                self.conv_out2 = ai8x.FusedConv1dBNReLU(96, 64, 3, stride=1, padding=0,
                        bias=bias, batchnorm='Affine', **kwargs)
                self.conv_out3 = ai8x.FusedConv1dBNReLU(64, 48, 3, stride=1, padding=0,
                        bias=bias, batchnorm='Affine', **kwargs)

                self.dropout = nn.Dropout(dropout)

                self.lin = ai8x.FusedLinearReLU(768, 128, bias=bias, **kwargs)

                ## MLP ##
                self.mlp1 = ai8x.FusedLinearReLU(128, 256, bias=bias, **kwargs)

                ## LINEAR ##
                self.fc_state = ai8x.Linear(256, num_classes*2, bias=bias, **kwargs)
                self.fc_power = ai8x.Linear(256, num_classes*5, bias=bias, **kwargs)

                self.initWeights("kaiming")

        def forward(self, x):
                                                        # 100
                x = self.prep(x)                        # 32x102

                ## 1ST ##
                x = self.conv0_1(x)                     # 32x102
                x_res = self.conv0_2(x)                 # 32x102
                x = self.conv0_3(x_res)                 # 32x102
                x = self.resid_0(x, x_res)              # 32x102
        
                # bottleneck residual block
                x_i = x
                x_i = self.downsample1(x_i)             # 96x49
                x = self.conv1_1(x)                     # 48x102
                x = self.conv1_2(x)                     # 64x49
                x = self.dropout1_1(x)                  # 96x49
                x = self.conv1_3(x)                     # 96x49
                x = self.resid_1(x, x_i)                # 96x49

                ## 2ND ##
                x = self.conv2_1(x)                     # 48x49
                x_res = self.conv2_2(x)                 # 48x22 (Conv+Pool)
                x = self.conv2_3(x_res)                 # 48x22
                x = self.resid_2(x, x_res)              # 48x22

                x = self.conv_out1(x)                   # 96x20
                x = self.conv_out2(x)                   # 64x18
                x = self.conv_out3(x)                   # 48x16
                x = self.dropout(x)
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