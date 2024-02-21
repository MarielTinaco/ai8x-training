import torch
import torch.nn.functional as F
from torch import nn

import ai8x

class AI85UNetNILM(nn.Module):
    """
    Large size UNet model. This model also enables the use of folded data.
    """
    def __init__(
            self,
            num_classes=4,         # in_size
            num_channels=48,        #
            dimensions=(88, 88),  # pylint: disable=unused-argument
            dropout=0.1,
            bias=True,
            **kwargs
    ):
        super().__init__()

        # self.fold_ratio = fold_ratio
        # self.num_classes = num_classes
        # self.num_final_channels = num_classes * fold_ratio * fold_ratio

        self.dropout = nn.Dropout(dropout)

        self.prep0 = ai8x.FusedConv1dBNReLU(num_channels, 64, 1, stride=1, padding=0,
                                            bias=bias, batchnorm='NoAffine', **kwargs)
        self.prep1 = ai8x.FusedConv1dBNReLU(64, 64, 1, stride=1, padding=0,
                                            bias=bias, batchnorm='NoAffine', **kwargs)
        self.prep2 = ai8x.FusedConv1dBNReLU(64, 32, 1, stride=1, padding=0,
                                            bias=bias, batchnorm='NoAffine', **kwargs)

        self.enc1 = ai8x.FusedConv1dBNReLU(32, 8, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)
        self.enc2 = ai8x.FusedMaxPoolConv1dBNReLU(8, 28, 3, stride=1, padding=1,
                                                  bias=bias, batchnorm='NoAffine', **kwargs)
        self.enc3 = ai8x.FusedMaxPoolConv1dBNReLU(28, 56, 3, stride=1, padding=1,
                                                  bias=bias, batchnorm='NoAffine', **kwargs)

        self.bneck = ai8x.FusedMaxPoolConv1dBNReLU(56, 112, 3, stride=1, padding=1,
                                                   bias=bias, batchnorm='NoAffine', **kwargs)

        self.upconv3 = ai8x.ConvTranspose2d(112, 56, 3, stride=2, padding=1)
        self.dec3 = ai8x.FusedConv1dBNReLU(112, 56, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)

        # self.upconv2 = nn.ConvTranspose1d(56, 28, 3, stride=2, padding=1)
        # self.dec2 = ai8x.FusedConv1dBNReLU(56, 28, 3, stride=1, padding=1,
        #                                    bias=bias, batchnorm='NoAffine', **kwargs)

        # self.upconv1 = nn.ConvTranspose1d(28, 8, 3, stride=2, padding=1)
        # self.dec1 = ai8x.FusedConv1dBNReLU(16, 48, 3, stride=1, padding=1,
        #                                    bias=bias, batchnorm='NoAffine', **kwargs)

        # self.dec0 = ai8x.FusedConv1dBNReLU(48, 64, 3, stride=1, padding=1,
        #                                    bias=bias, batchnorm='NoAffine', **kwargs)

        # self.conv_p1 = ai8x.FusedConv1dBNReLU(64, 64, 1, stride=1, padding=0,
        #                                       bias=bias, batchnorm='NoAffine', **kwargs)
        # self.conv_p2 = ai8x.FusedConv1dBNReLU(64, 64, 1, stride=1, padding=0,
        #                                       bias=bias, batchnorm='NoAffine', **kwargs)
        # self.conv_p3 = ai8x.Conv1d(64, 64, 1, stride=1, padding=0,
        #                                   bias=bias, batchnorm='NoAffine', **kwargs)

        # self.conv = ai8x.Conv1d(64, self.num_final_channels, 1, stride=1, padding=0,
        #                             bias=bias, batchnorm='NoAffine', **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        # Run CNN
        B = x.size(0)
        x = x.permute(0,2,1)

        x = self.prep0(x)
        # x = self.prep1(x)
        # x = self.prep2(x)
        
        # # Encoder
        # enc1 = self.enc1(x)                    # 8x(dim1)x(dim2)
        # enc2 = self.enc2(enc1)                 # 28x(dim1/2)x(dim2/2)
        # enc3 = self.enc3(enc2)                 # 56x(dim1/4)x(dim2/4)

        # bottleneck = self.bneck(enc3)          # 112x(dim1/8)x(dim2/8)

        # dec3 = self.upconv3(bottleneck)        # 56x(dim1/4)x(dim2/4)
        # dec3 = torch.cat((dec3, enc3), dim=1)  # 112x(dim1/4)x(dim2/4)
        # dec3 = self.dec3(dec3)                 # 56x(dim1/4)x(dim2/4)
        # dec2 = self.upconv2(dec3)              # 28x(dim1/2)x(dim2/2)
        # dec2 = torch.cat((dec2, enc2), dim=1)  # 56(dim1/2)x(dim2/2)
        # dec2 = self.dec2(dec2)                 # 28x(dim1/2)x(dim2/2)
        # dec1 = self.upconv1(dec2)              # 8x(dim1)x(dim2)
        # dec1 = torch.cat((dec1, enc1), dim=1)  # 16x(dim1)x(dim2)
        # dec1 = self.dec1(dec1)                 # 48x(dim1)x(dim2)
        # dec0 = self.dec0(dec1)                 # 64x(dim1)x(dim2)

        # dec0 = self.conv_p1(dec0)
        # dec0 = self.conv_p2(dec0)
        # dec0 = self.conv_p3(dec0)
        # dec0 = self.conv(dec0)                 # num_final_channelsx(dim1)x(dim2)

        # return dec0

        x = self.dropout(x)

        return x

class AI85CNN1DNiLM(nn.Module):
    def __init__(self, in_size=1, 
                 output_size=5,
                 d_model=64,
                 dropout=0.1, 
                 seq_len=99,
                 n_layers=5, 
                 n_quantiles=3, 
                 pool_filter=16,
                 device="cuda:0"):
        super(AI85CNN1DNiLM, self).__init__()

        self.enc_net = Encoder(n_channels=in_size, n_kernels=d_model, n_layers=n_layers, seq_size=seq_len, device=device)
        self.pool_filter = pool_filter
        self.mlp_layer = MLPLayer(in_size=d_model*pool_filter, hidden_arch=[1024], output_size=None)
        self.dropout = nn.Dropout(dropout)
        self.n_quantiles = n_quantiles
        
        self.fc_out_state  = ai8x.Linear(1024, output_size*2, bias=True)
        self.fc_out_power  = ai8x.Linear(1024, output_size*n_quantiles, bias=True)
        nn.init.xavier_normal_(self.fc_out_state.op.weight)
        nn.init.xavier_normal_(self.fc_out_power.op.weight)
        
    def forward(self, x):
        x = x.permute(0,2,1)
        B = x.size(0)
        conv_out = self.dropout(self.enc_net(x))
        conv_out = F.adaptive_avg_pool1d(conv_out, self.pool_filter).reshape(x.size(0), -1)
        mlp_out  = self.dropout(self.mlp_layer(conv_out))
        states_logits   = self.fc_out_state(mlp_out).reshape(B, 2, -1)
        power_logits    = self.fc_out_power(mlp_out)
        if self.n_quantiles>1:
            power_logits = power_logits.reshape(B, self.n_quantiles, -1)
            
        return states_logits, power_logits

class Encoder(nn.Module):
    def __init__(self, 
                 n_channels=10, 
                 n_kernels=16, 
                 n_layers=3, 
                 seq_size=50,
                 device="cuda:0"):
        super(Encoder, self).__init__()
        self.feat_size = (seq_size-1) // 2**n_layers +1
        self.feat_dim = self.feat_size * n_kernels
        self.conv_stack = nn.Sequential(
            *([Conv1D(n_channels, n_kernels // 2**(n_layers-1), activation="ReLU", pooling="Max", last=False, device=device)] +
              [Conv1D(n_kernels//2**(n_layers-l),
                         n_kernels//2**(n_layers-l-1), activation="ReLU", pooling="Max", last=False, device=device)
               for l in range(1, n_layers-1)] +
              [Conv1D(n_kernels // 2, n_kernels, activation="ReLU", pooling="Max", last=True, device=device)])
        )
    def forward(self, x):
        assert len(x.size())==3
        feats = self.conv_stack(x)
        return feats

class MLPLayer(nn.Module):
    def __init__(self, in_size, 
                 hidden_arch=[128/2, 512/2, 1024/2], 
                 output_size=None):
        
        super(MLPLayer, self).__init__()
        self.in_size = in_size
        self.output_size = output_size
        layer_sizes = [in_size] + [x for x in hidden_arch]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = ai8x.FusedLinearReLU(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)

        if output_size is not None:
            layer = ai8x.FusedLinearReLU(layer_sizes[-1], output_size)
            self.layers.append(layer)

        self.init_weights()
        self.mlp_network =  nn.Sequential(*self.layers)

    def forward(self, z):
        return self.mlp_network(z)
        
    def init_weights(self):
        for layer in self.layers:
            try:
                if isinstance(layer, ai8x.FusedLinearReLU):
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