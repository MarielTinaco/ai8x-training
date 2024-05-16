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


class AI85NILMSeq2PointRegress(nn.Module):
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

		dropout = 0.25

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

		self.conv6 = ai8x.FusedAvgPoolConv1dBNReLU(50, 64, 4, stride=1, padding=0,
						bias=bias, batchnorm='Affine', **kwargs)

		self.fc_state = ai8x.Linear(128, num_classes*2, bias=bias, wide=True, **kwargs)
		self.fc_power = ai8x.Linear(128, num_classes*5, bias=bias, wide=True, **kwargs)

		self.initWeights("kaiming")

	def forward(self, x):
		B = x.size(0)

		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = self.conv6(x)
		x = self.dropout(x)
		x = x.view(x.size(0), -1)
		x1 = self.fc_state(x).reshape(B, 2, -1)
		x2 = self.fc_power(x).reshape(B, 5, -1)
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

def ai85nilmseq2pointregress(pretrained=False, **kwargs):
    """
    Constructs a AI85KWS20Net model.
    """
    assert not pretrained
    return AI85NILMSeq2PointRegress(**kwargs)


models = [
    {
        'name': 'ai85nilmseq2pointregress',
        'min_input': 1,
        'dim': 1,
    },
]