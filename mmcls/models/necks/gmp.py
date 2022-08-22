# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import NECKS

@NECKS.register_module()
class GlobalMaxPooling(nn.Module):
    """Global MAX Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """
    def __init__(self, dim=2):
        super(GlobalMaxPooling, self).__init__()
        assert dim in [1, 2, 3], 'GlobalAveragePooling dim only support ' \
            f'{1, 2, 3}, get {dim} instead.'

        if dim == 1:
            self.gmp = nn.AdaptiveMaxPool1d(1)
        elif dim == 2:
            self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.gmp = nn.AdaptiveMaxPool3d((1, 1, 1))

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.gmp(x) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gmp(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs

class unit_tcn(nn.Module):
    def __init__(self, in_channels=2560, out_channels=2560, bias=True):
        super(unit_tcn, self).__init__()

        self.maxpool = nn.AdaptiveMaxPool2d((1, 7))

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.dropout = nn.Dropout2d(0.2)

        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        

    def forward(self, x):
        r"""
        shape of x: (N, 256, V, T)
        """
        x = self.maxpool(x) # shape (N, 256, 1, 20)

        x = self.conv_1(x) # shape (N, 256, 1, 20)
        x = self.dropout(x)
        x = self.conv_2(x) # shape (N, 512, 1, 20)

        return x

@NECKS.register_module()
class UnitTCN(nn.Module):
    """Global MAX Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """
    def __init__(self, dim=2):
        super(UnitTCN, self).__init__()
        assert dim in [1, 2, 3], 'GlobalAveragePooling dim only support ' \
            f'{1, 2, 3}, get {dim} instead.'

        self.tcn = unit_tcn()

        if dim == 1:
            self.gmp = nn.AdaptiveMaxPool1d(1)
        elif dim == 2:
            self.gmp = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.gmp = nn.AdaptiveMaxPool3d((1, 1, 1))

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            #outs = tuple([self.gmp(x) for x in inputs])
            outs = tuple([self.gmp(self.tcn(x)) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gmp(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
