###################################################################################################
#
# Copyright (C) 2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
SimpleNet_v1 network with added residual layers for AI85.
Simplified version of the network proposed in [1].

[1] HasanPour, Seyyed Hossein, et al. "Lets keep it simple, using simple architectures to
    outperform deeper and more complex architectures." arXiv preprint arXiv:1608.06037 (2016).
"""
from torch import nn

import torch
import ai8x


class AI85ResidualSimpleNet(nn.Module):
    """
    Residual SimpleNet v1 Model
    """
    def __init__(
            self,
            num_classes=5,
            num_channels=1,
            dimensions=(20, 20),  # pylint: disable=unused-argument
            bias=False,
            **kwargs
    ):
        super().__init__()
        dropout = 0.25

        self.dropout = nn.Dropout(dropout)
        ## INIT ##
        self.conv1 = ai8x.FusedConv2dBNReLU(num_channels, 20, 3, stride=1, padding=1,
                bias=bias, batchnorm='Affine', **kwargs)

        ## 1ST ##
        self.conv2 = ai8x.FusedConv2dBNReLU(20, 32, 3, stride=1, padding=1,
                bias=bias, batchnorm='Affine', **kwargs)

        self.conv3 = ai8x.FusedConv2dBNReLU(32, 32, 3, stride=1, padding=1,
                bias=bias, batchnorm='Affine', **kwargs)
        self.resid_1 = ai8x.Add()

        ## 2ND ##
        self.conv4 = ai8x.FusedConv2dBNReLU(32, 32, 3, stride=1, padding=1,
                bias=bias, batchnorm='Affine', **kwargs)
        # residual max pool 2
        self.conv5_max = ai8x.FusedMaxPoolConv2dBNReLU(32, 48, 3, stride=1, padding=1,
                bias=bias, batchnorm='Affine', **kwargs)
        self.conv6 = ai8x.FusedConv2dBNReLU(48, 48, 3, stride=1, padding=1,
                bias=bias, batchnorm='Affine', **kwargs)
        self.resid_2 = ai8x.Add()

        ## 3RD ##
        self.conv7 = ai8x.FusedConv2dBNReLU(48, 48, 3, stride=1, padding=1,
                bias=bias, batchnorm='Affine', **kwargs)
        # residual max pool 3
        self.conv8_max = ai8x.FusedMaxPoolConv2dBNReLU(48, 64, 3, stride=1, padding=1,
                bias=bias, batchnorm='Affine', **kwargs)
        self.conv9 = ai8x.FusedConv2dBNReLU(64, 64, 3, stride=1, padding=1,
                bias=bias, batchnorm='Affine', **kwargs)
        self.resid_3 = ai8x.Add()

        ## 4TH ##
        self.conv10 = ai8x.FusedConv2dBNReLU(64, 64, 3, stride=1, padding=1,
                bias=bias, batchnorm='Affine', **kwargs)
        self.conv11_max = ai8x.FusedMaxPoolConv2dBNReLU(64, 64, 3, stride=1, padding=1,
                bias=bias, batchnorm='Affine', **kwargs)
        self.conv12 = ai8x.FusedConv2dBNReLU(64, 64, 3, stride=1, padding=1,
                bias=bias, batchnorm='Affine', **kwargs)
        self.resid_4 = ai8x.Add()

        # ## EXTRA ##

        self.conv13 = ai8x.FusedConv2dBNReLU(64, 64, 3, stride=1, padding=1,
                bias=bias, batchnorm='Affine', **kwargs)

        self.conv14 =  ai8x.FusedMaxPoolConv2dBNReLU(64, 64, 3, stride=1, padding=1,
                bias=bias, batchnorm='Affine', **kwargs)
        
#         self.conv15 =  ai8x.FusedMaxPoolConv2dBNReLU(128, 128, 3, stride=1, padding=1,
#                 bias=bias, batchnorm='Affine', **kwargs)

        self.mlp1 = ai8x.FusedLinearReLU(64, 256, bias=bias, **kwargs)

#         self.mlp2 = ai8x.FusedLinearReLU(256, 256, bias=bias, **kwargs)

        self.fc_state = ai8x.Linear(256, num_classes*2, bias=bias, **kwargs)
        self.fc_power = ai8x.Linear(256, num_classes*5, bias=bias, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        #                               # Conv: [(W−K+2P)/S]+1
                                        # Pool: [(W-K)/S] + 1
        ## INIT ##                      # 1x20x20
        x = self.conv1(x)               # 20x20x20

        ## 1ST ##
        x_res = self.conv2(x)           # 32x20x20
        x = self.conv3(x_res)           # 32x20x20
        x = self.resid_1(x, x_res)      # 32x20x20

        ## 2ND ##
        x = self.conv4(x)               # 32x20x20
        x_res = self.conv5_max(x)       # 48x10x10 (Conv+Pool)
        x = self.conv6(x_res)           # 48x10x10
        x = self.resid_2(x, x_res)      # 48x10x10

        ## 3RD ##
        x = self.conv7(x)               # 48x10x10
        x_res = self.conv8_max(x)       # 64x5x5 (Conv+Pool)
        x = self.conv9(x_res)           # 64x5x5
        x = self.resid_3(x, x_res)      # 64x5x5

        ## 4TH ##
        x = self.conv10(x)              # 64x5x5
        x_res = self.conv11_max(x)      # 64x2x2 (Conv+Pool+rounddown)
        x = self.conv12(x_res)          # 64x2x2
        x = self.resid_4(x, x_res)      # 64x2x2

        ## EXTRA ##
        x = self.conv13(x)              # 64x2x2
        x = self.conv14(x)              # 64x1x1
        x = self.dropout(x)             # 64x1x1
        x = x.view(x.size(0), -1)       # 320
        x = self.mlp1(x)                # 256x1
        # x = self.mlp2(x)
        x1 = self.fc_state(x)           # 10
        x1 = x1.view(x1.size(0), -1)    # 10
        x2 = self.fc_power(x)           # 25
        x2 = x2.view(x2.size(0), -1)    # 25
        return torch.cat([x1, x2], dim=1) # 35


def ai85nilmslidingwindowressimplenet(pretrained=False, **kwargs):
    """
    Constructs a Residual SimpleNet v1 model.
    """
    assert not pretrained
    return AI85ResidualSimpleNet(**kwargs)


models = [
    {
        'name': 'ai85nilmslidingwindowressimplenet',
        'min_input': 1,
        'dim': 2,
    },
]
