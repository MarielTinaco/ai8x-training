###################################################################################################
#
# Copyright (C) 2023-2024 Analog Devices, Inc. All Rights Reserved.
#
# Analog Devices, Inc. Default Copyright Notice:
# https://www.analog.com/en/about-adi/legal-and-risk-oversight/intellectual-property/copyright-notice.html
#
###################################################################################################
#
# Copyright (C) 2021-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Keyword spotting network for AI85
"""
from torch import nn

import ai8x


class AI85NILMNetNAS(nn.Module):
    """
    KWS20 NAS Audio net, found via Neural Architecture Search
    It significantly outperforms earlier networks (v1, v2, v3), though with a higher
    parameter count and slightly increased latency.
    """

    # num_classes = n keywords + 1 unknown
    def __init__(
            self,
            num_classes=5,
            num_channels=100,
            dimensions=(128, 1),  # pylint: disable=unused-argument
            bias=True,
            **kwargs
    ):
        super().__init__()
        # T: 128 F :128
        self.conv1 = ai8x.FusedConv1dBNReLU(num_channels, 100, 1, stride=1, padding=0,
                                            bias=bias, batchnorm='Affine', **kwargs)
        # T:  128 F: 100
        self.conv2 = ai8x.FusedConv1dBNReLU(100, 48, 1, stride=1, padding=1,
                                            bias=bias, batchnorm='Affine', **kwargs)
        # T: 126 F : 48
        self.conv3 = ai8x.FusedMaxPoolConv1dBNReLU(48, 96, 1, stride=1, padding=1,
                                                   bias=bias, batchnorm='Affine', **kwargs)
        # T: 62 F : 96
        self.conv4 = ai8x.FusedConv1dBNReLU(96, 128, 1, stride=1, padding=0,
                                            bias=bias, batchnorm='Affine', **kwargs)
        # T : 60 F : 128
        self.conv5 = ai8x.FusedMaxPoolConv1dBNReLU(128, 160, 1, stride=1, padding=1,
                                                   bias=bias, batchnorm='Affine', **kwargs)
        # T: 30 F : 160
        self.conv6 = ai8x.FusedConv1dBNReLU(160, 192, 1, stride=1, padding=0,
                                            bias=bias, batchnorm='Affine', **kwargs)
        # T: 28 F : 192
        self.conv7 = ai8x.FusedAvgPoolConv1dBNReLU(192, 192, 1, stride=1, padding=1,
                                                   bias=bias, batchnorm='Affine', **kwargs)
        # T : 14 F: 256
        self.conv8 = ai8x.FusedConv1dBNReLU(192, 32, 1, stride=1, padding=0,
                                            bias=bias, batchnorm='Affine', **kwargs)
        self.fc_state = ai8x.Linear(96, num_classes*2, bias=bias, wide=True, **kwargs)
        self.fc_power = ai8x.Linear(96, num_classes*5, bias=bias, wide=True, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        # Run CNN
        B = x.size(0)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = x.view(x.size(0), -1)
        x1 = self.fc_state(x).reshape(B, 2, -1)
        x2 = self.fc_power(x).reshape(B, 5, -1)
        return (x1, x2)


def ai85nilmnetnas(pretrained=False, **kwargs):
    """
    Constructs a AI85KWS20NetNAS model.
    """
    assert not pretrained
    return AI85NILMNetNAS(**kwargs)


models = [
    {
        'name': 'ai85nilmnetnas',
        'min_input': 1,
        'dim': 1,
    },
]
