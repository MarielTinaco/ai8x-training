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


class AI85NILMAutoEncoderRegress(nn.Module):
    """
    Small size UNet model
    """
    def __init__(
            self,
            num_classes=5,
            num_channels=1,
            dimensions=(100, 1),  # pylint: disable=unused-argument
            bias=True,
            weight_init="kaiming",
            batchNorm=True,
            bottleNeckDim=4,
            **kwargs
    ):
        super().__init__()

        print("Batchnorm setting in model = ", batchNorm)

        weight_init = weight_init.lower()
        assert weight_init in ('kaiming', 'xavier', 'glorot')

        # Num channels is equal to the length of FFTs here
        self.num_channels = num_channels
        self.n_axes =  1

        S = 1
        P = 0

        # ----- DECODER ----- #
        # Kernel in 1st layer looks at 1 axis at a time. Output width = input width
        n_in = num_channels
        n_out = 128
        if batchNorm:
            self.en_conv1 = ai8x.FusedConv1dBNReLU(n_in, n_out, 1, stride=S, padding=P, dilation=1,
                                                   bias=bias, batchnorm='Affine', **kwargs)
        else:
            self.en_conv1 = ai8x.FusedConv1dReLU(n_in, n_out, 1, stride=S, padding=P, dilation=1,
                                                 bias=bias, **kwargs)
        self.layer1_n_in = n_in
        self.layer1_n_out = n_out

        # Kernel in 2nd layer looks at 3 axes at once. Output Width = 1. Depth=n_out
        n_in = n_out
        n_out = 64
        if batchNorm:
            self.en_conv2 = ai8x.FusedConv1dBNReLU(n_in, n_out, 1, stride=S, padding=P, dilation=1,
                                                   bias=bias, batchnorm='Affine', **kwargs)
        else:
            self.en_conv2 = ai8x.FusedConv1dReLU(n_in, n_out, 1, stride=S, padding=P, dilation=1,
                                                 bias=bias, **kwargs)
        self.layer2_n_in = n_in
        self.layer2_n_out = n_out

        n_in = n_out
        n_out = 32
        self.en_lin1 = ai8x.FusedLinearReLU(n_in, n_out, bias=bias, **kwargs)
        # ----- END OF DECODER ----- #

        # ---- BOTTLENECK ---- #
        n_in = n_out
        self.bottleNeckDim = bottleNeckDim
        n_out = self.bottleNeckDim
        self.en_lin2 = ai8x.Linear(n_in, n_out, bias=0, **kwargs)
        # ---- END OF BOTTLENECK ---- #

        # ----- ENCODER ----- #
        n_in = n_out
        n_out = 32
        self.de_lin1 = ai8x.FusedLinearReLU(n_in, n_out, bias=bias, **kwargs)

        n_in = n_out
        n_out = 96
        self.de_lin2 = ai8x.FusedLinearReLU(n_in, n_out, bias=bias, **kwargs)

        n_in = n_out
        n_out = num_channels* 1
        self.de_lin3 = ai8x.FusedLinearReLU(n_in, n_out, bias=bias, **kwargs)

        n_in = n_out

        # --- SEQ2POINT --- #
        self.conv1 = ai8x.FusedConv1dBNReLU(self.n_axes, 30, 9, stride=1, padding=0,
											bias=bias, batchnorm='Affine', **kwargs)

        self.conv2 = ai8x.FusedConv1dBNReLU(30, 30, 8, stride=1, padding=0,
											bias=bias, batchnorm='Affine', **kwargs)

        self.conv3 = ai8x.FusedMaxPoolConv1dBNReLU(30, 40, 6, stride=1, padding=0,
											bias=bias, batchnorm='Affine', **kwargs)
		
        self.conv4 = ai8x.FusedMaxPoolConv1dBNReLU(40, 50, 5, stride=1, padding=0,
											bias=bias, batchnorm='Affine', **kwargs)
		
        self.conv5 = ai8x.FusedConv1dBNReLU(50, 50, 5, stride=1, padding=0,
											bias=bias, batchnorm='Affine', **kwargs)

        self.conv6 = ai8x.FusedConv1dBNReLU(50, 50, 5, stride=1, padding=0,
											bias=bias, batchnorm='Affine', **kwargs)
		
        self.out_lin_state = ai8x.Linear(300, num_classes * 2, bias=0, **kwargs)
        self.out_lin_power = ai8x.Linear(300, num_classes * 5, bias=0, **kwargs)
        # ----- END OF ENCODER ----- #

        self.initWeights(weight_init)

    def forward(self, x, return_bottleneck=False):
        """Forward prop"""
        B = x.size(0)

        x = self.en_conv1(x)
        x = self.en_conv2(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.en_lin1(x)
        x = self.en_lin2(x)

        if return_bottleneck:
            return x

        x = self.de_lin1(x)
        x = self.de_lin2(x)
        x = self.de_lin3(x)

        x = x.view(x.shape[0], self.n_axes, x.shape[1])

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = x.view(x.shape[0], -1)

        x1 = self.out_lin_state(x).reshape(B, 2, -1)
        x2 = self.out_lin_power(x).reshape(B, 5, -1)
        # x = x.view(x.shape[0], self.num_channels, self.n_axes)

        return (x1, x2)


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


def ai85nilmautoencoderregress(pretrained=False, **kwargs):
    """
    Constructs a AI85KWS20Net model.
    """
    assert not pretrained
    return AI85NILMAutoEncoderRegress(**kwargs)


models = [
    {
        'name': 'ai85nilmautoencoderregress',
        'min_input': 1,
        'dim': 1,
    },
]