import mmcv
import numpy as np

from .builder import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class Orange21(BaseDataset):

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(';') for x in f.readlines()]
            for filename, gt_label, bbox, scale in samples:
                info = {'img_prefix': self.data_prefix}
                #info['img_info'] = {'filename': '{}.jpg'.format(filename)}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)

                # process bbox from str to float
                bbox = bbox.replace('[', '').replace(']', '')
                bbox = [float(pos) for pos in bbox.split(',')]
                info['bbox'] = np.array(bbox, dtype=np.int64)

                # process depth scale from str to float
                info['scale'] = np.float32(scale)
                
                data_infos.append(info)
            return data_infos


@DATASETS.register_module()
class OrangeDiameter(BaseDataset):

    def load_annotations(self):
        assert isinstance(self.ann_file, str)
        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(';') for x in f.readlines()]
            for filename, gt_label, bbox, scale in samples:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)

                # process bbox from str to float
                bbox = bbox.replace('[', '').replace(']', '')
                bbox = [float(pos) for pos in bbox.split(',')]
                info['bbox'] = np.array(bbox, dtype=np.int64)

                # process depth scale from str to float
                info['scale'] = np.float32(scale)

                data_infos.append(info)
            return data_infos

@DATASETS.register_module()
class OrangeGrade(BaseDataset):

    def load_annotations(self):
        assert isinstance(self.ann_file, str)
        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(';') for x in f.readlines()]
            for filename, gt_label, bbox, scale in samples:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)

                # process bbox from str to float
                bbox = bbox.replace('[', '').replace(']', '')
                bbox = [float(pos) for pos in bbox.split(',')]
                info['bbox'] = np.array(bbox, dtype=np.int64)

                # process depth scale from str to float
                info['scale'] = np.float32(scale)

                data_infos.append(info)
            return data_infos






