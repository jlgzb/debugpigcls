import torch
import torch.nn as nn

from ..builder import NECKS

#from mmcv.cnn import build_conv_layer, build_norm_layer

class unit_adjacent(nn.Module):
    def __init__(self):
        super(unit_adjacent, self).__init__()
        self.conv_query = nn.Conv2d(in_channels=2304, out_channels=2304, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels=2304, out_channels=2304, kernel_size=1)
        self.softmax = nn.Softmax(-1)

    def forward(self, input):
        '''
        shape:
            input: (N, C, V, T)
            output: (N, C, V, T)
        '''
        query = self.conv_query(input) # (N, C, V, T)
        key = self.conv_key(input).permute(0, 1, 3, 2).contiguous() # (N, C, T, V)
        A = torch.matmul(query, key) # (N, C, V, V)
        return self.softmax(A)

class unit_gcn(nn.Module):
    def __init__(self, in_C=2304, out_C=512):
        super(unit_gcn, self).__init__()
        
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels=in_C, out_channels=out_C, kernel_size=1),
            nn.BatchNorm2d(out_C),
        )

        self.conv1 = nn.Conv2d(in_channels=in_C, out_channels=out_C, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_C)

        self.relu = nn.ReLU()

    def forward(self, input, A):
        ''' spatial conv? 
        shape:
            input: (N, C, V, T)
            A: (N, C, V, V) or (N, C, T, T)
        '''

        out = torch.matmul(A, input) # (N, C, V, T)
        
        output = self.bn(self.conv1(out)) + self.residual(input)
        output = self.relu(output)
        return output

@NECKS.register_module()
class UnitGCNGAPEff(nn.Module):
    """GCN.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2

    20220328
    shape:
        input: (-1, 2560, 7, 7)
        output: (-1, -1)
    """
    def __init__(self, dim=2):
        super(UnitGCNGAPEff, self).__init__()
        assert dim in [1, 2, 3], 'GlobalAveragePooling dim only support ' \
            f'{1, 2, 3}, get {dim} instead.'

        self.adjacent = unit_adjacent()
        self.gcn1 = unit_gcn()
        
        if dim == 1:
            self.gmp = nn.AdaptiveAvgPool1d(1)
        elif dim == 2:
            self.gmp = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.gmp = nn.AdaptiveAvgPool3d((1, 1, 1))
        
    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.gmp(self.gcn1(x, self.adjacent(x))) for x in inputs])
            outs = tuple([out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gmp(self.gcn1(inputs, self.adjacent(inputs)))
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs

@NECKS.register_module()
class UnitGCNGMPEff(nn.Module):
    """GCN.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2

    20220328
    shape:
        input: (-1, 2560, 7, 7)
        output: (-1, -1)
    """
    def __init__(self, dim=2):
        super(UnitGCNGMPEff, self).__init__()
        assert dim in [1, 2, 3], 'GlobalAveragePooling dim only support ' \
            f'{1, 2, 3}, get {dim} instead.'

        self.adjacent = unit_adjacent()
        self.gcn1 = unit_gcn()
        
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
            outs = tuple([self.gmp(self.gcn1(x, self.adjacent(x))) for x in inputs])
            outs = tuple([out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gmp(self.gcn1(inputs, self.adjacent(inputs)))
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
