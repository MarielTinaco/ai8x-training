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


class AI85NILMSeq2Point(nn.Module):
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

		self.dropout = nn.Dropout(dropout)

		self.conv1 = ai8x.FusedConv1dBNReLU(num_channels, 30, 9, stride=1, padding=0,
											bias=bias, batchnorm='Affine', **kwargs)
		
		self.conv2 = ai8x.FusedConv1dBNReLU(30, 30, 8, stride=1, padding=0,
											bias=bias, batchnorm='Affine', **kwargs)
		
		self.conv3 = ai8x.FusedMaxPoolConv1dBNReLU(30, 40, 6, stride=1, padding=0,
											bias=bias, batchnorm='Affine', **kwargs)
		
		self.conv4 = ai8x.FusedMaxPoolConv1dBNReLU(40, 50, 5, stride=1, padding=0,
											bias=bias, batchnorm='Affine', **kwargs)
		
		self.conv5 = ai8x.FusedConv1dBNReLU(50, 50, 5, stride=1, padding=0,
											bias=bias, batchnorm='Affine', **kwargs)
		
		self.mlp = ai8x.FusedLinearReLU(500, 256, bias=bias, **kwargs)

		self.fc = ai8x.Linear(256, num_classes*2, bias=bias, wide=True, **kwargs)


	def forward(self, x):
		B = x.size(0)

		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		
		x = self.dropout(x)
		x = x.view(x.size(0), -1)
		x = self.mlp(x)
		x = self.fc(x)
		return x.reshape(B, 2, -1)

def ai85nilmseq2point(pretrained=False, **kwargs):
    """
    Constructs a AI85KWS20Net model.
    """
    assert not pretrained
    return AI85NILMSeq2Point(**kwargs)


models = [
    {
        'name': 'ai85nilmseq2point',
        'min_input': 1,
        'dim': 1,
    },
]