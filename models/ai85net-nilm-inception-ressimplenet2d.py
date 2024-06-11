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


class AI85NILM2DResidualSimplenet(nn.Module):
        def __init__(
                self,
                num_classes=5,
                num_channels=1,
                dimensions=(100, 1),  # pylint: disable=unused-argument
                bias=True,
                **kwargs
        ):
                super().__init__()

                dropout = 0.25

                self.dropout = nn.Dropout(dropout)

                self.prep1 = ai8x.FusedConv1dBNReLU(num_channels, 64, 3, stride=1, padding=1,
                        bias=bias, batchnorm='Affine', **kwargs)

                self.inc_branch0_0 = ai8x.FusedConv1dBNReLU(64, 16, 1, stride=1, padding=0,
                        bias=bias, batchnorm='Affine', **kwargs)

                self.inc_branch1_0 = ai8x.FusedConv1dBNReLU(64, 32, 1, stride=1, padding=0,
                        bias=bias, batchnorm='Affine', **kwargs)
                self.inc_branch1_1 = ai8x.FusedConv1dBNReLU(32, 16, 3, stride=1, padding=1,
                        bias=bias, batchnorm='Affine', **kwargs)

                self.inc_branch2_0 = ai8x.FusedConv1dBNReLU(64, 32, 1, stride=1, padding=0,
                        bias=bias, batchnorm='Affine', **kwargs)
                self.inc_branch2_1 = ai8x.FusedConv1dBNReLU(32, 16, 5, stride=1, padding=2,
                        bias=bias, batchnorm='Affine', **kwargs)
                
                self.inc_branch3_0 = ai8x.FusedMaxPoolConv1dBNReLU(64, 16, 1, stride=1, padding=1,
                        pool_stride=1, pool_size=3)

                # ResNet
                self.res_conv1 = ai8x.FusedConv2dReLU(64, 16, 3, stride=1, padding=1, bias=bias,
                                                **kwargs)
                self.res_conv2 = ai8x.FusedConv2dReLU(16, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
                self.res_conv3 = ai8x.FusedConv2dReLU(20, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
                self.res_conv4 = ai8x.FusedConv2dReLU(20, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
                self.resid1 = ai8x.Add()

                self.res_conv5 = ai8x.FusedMaxPoolConv2dReLU(20, 20, 3, pool_size=2, pool_stride=2,
                                                 stride=1, padding=1, bias=bias, **kwargs)
                self.res_conv6 = ai8x.FusedConv2dReLU(20, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
                self.resid2 = ai8x.Add()
                self.res_conv7 = ai8x.FusedConv2dReLU(20, 32, 3, stride=1, padding=1, bias=bias, **kwargs)

                self.mlp1 = ai8x.FusedLinearReLU(800, 256, bias=bias, **kwargs)

                self.fc_state = ai8x.Linear(256, num_classes*2, bias=bias, **kwargs)
                self.fc_power = ai8x.Linear(256, num_classes*5, bias=bias, **kwargs)

                self.initWeights("kaiming")

        def forward(self, x):

                x = self.prep1(x)

                x_0 = self.inc_branch0_0(x)

                x_1 = self.inc_branch1_0(x)
                x_1 = self.inc_branch1_1(x_1)

                x_2 = self.inc_branch2_0(x)
                x_2 = self.inc_branch2_1(x_2)

                x_3 = self.inc_branch3_0(x)

                x = torch.cat((x_0, x_1, x_2, x_3), dim=1)

                x = x.view(x.shape[0], x.shape[1], 10, -1)      # 96 x 10 x 10

                x = self.res_conv1(x)          # 16x10x10
                x_res = self.res_conv2(x)      # 20x10x10
                x = self.res_conv3(x_res)      # 20x10x10
                x = self.resid1(x, x_res)      # 20x10x10
                x = self.res_conv4(x)          # 20x10x10

                x_res = self.res_conv5(x)      # 20x5x5
                x = self.res_conv6(x_res)      # 20x5x5
                x = self.resid2(x, x_res)      # 20x5x5
                x = self.res_conv7(x)          # 44x5x5

                x = x.view(x.shape[0], -1)

                x = self.mlp1(x)
                x1 = self.fc_state(x)
                x1 = x1.view(x1.size(0), -1)
                x2 = self.fc_power(x)
                x2 = x2.view(x2.size(0), -1)
                return torch.cat([x1, x2], dim=1)

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

def ai85nilminception2dressimplenet(pretrained=False, **kwargs):
        """
        Constructs a AI85KWS20Net model.
        """
        assert not pretrained
        return AI85NILM2DResidualSimplenet(**kwargs)


models = [
        {
                'name': 'ai85nilminception2dressimplenet',
                'min_input': 1,
                'dim': 1,
        },
]