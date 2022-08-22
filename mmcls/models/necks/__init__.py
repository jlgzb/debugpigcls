# Copyright (c) OpenMMLab. All rights reserved.
from .gap import GlobalAveragePooling
from .gem import GeneralizedMeanPooling
from .hr_fuse import HRFuseScales
from .gmp import GlobalMaxPooling, UnitTCN
from .gcn import UnitGCN, UnitGCNTwo, UnitGCNThree
from .gcn import UnitGCNGAP, UnitGCNGMP
from .gcn_efficient import UnitGCNGAPEff, UnitGCNGMPEff
from .gcn_hrnet import UnitGCNGAPHrn, UnitGCNGMPHrn

__all__ = [
    'GlobalAveragePooling', 'GeneralizedMeanPooling', 'HRFuseScales',
    'GlobalMaxPooling', 'UnitGCN', 'UnitGCNTwo', 'UnitGCNThree', 'UnitTCN',
    'UnitGCNGAP', 'UnitGCNGMP',
    'UnitGCNGAPEff', 'UnitGCNGMPEff', 'UnitGCNGAPHrn', 'UnitGCNGMPHrn'
]
