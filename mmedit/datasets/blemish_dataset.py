# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import torch
import numpy as np
from .base_generation_dataset import BaseGenerationDataset
from .registry import DATASETS
from collections import defaultdict
from mmedit.core.registry import build_metric
import torch.nn.functional as F
from pytorch_msssim import ssim as s_score

FEATURE_BASED_METRICS = ['PSNR', 'SSIM']

@DATASETS.register_module()
class BlemishDataset(BaseGenerationDataset):
    def __init__(self, dataroot, pipeline, test_mode=False, root_name=dict(), mask_skin='', Voxel_S=0.98, Voxel_Y=6.1):
        super().__init__(pipeline, test_mode)
        self.Voxel_S = Voxel_S
        self.Voxel_Y = Voxel_Y
        self.dataroot_list = dataroot
        self.data_infos = []
        self.mask_skin = mask_skin
        for i in range(len(self.dataroot_list)):
            self.dataroot = self.dataroot_list[i]
            self.name = root_name
            self.data_infos += self.load_annotations()

    def load_annotations(self):

        data_infos = []
        dataroot_a = osp.join(self.dataroot, self.name['A'])
        dataroot_b = osp.join(self.dataroot, self.name['B'])

        a_paths = self.scan_folder(dataroot_a)
        b_paths = self.scan_folder(dataroot_b)

        a_paths = sorted(a_paths)
        b_paths = sorted(b_paths)
        for apath, bpath in zip(a_paths, b_paths):
            assert osp.basename(apath) == osp.basename(bpath)
            mask_skin_path = osp.join(self.mask_skin, osp.basename(apath))
            mask_skin_path = mask_skin_path.replace('.jpg', '.png')
            data_infos.append(
                dict(A_path=apath, B_path=bpath, mask_skin_path=mask_skin_path))

        return data_infos

    def evaluate(self, results, logger=None):
        """Evaluating with saving generated images. (needs no metrics)

        Args:
            results (list[tuple]): The output of forward_test() of the model.

        Return:
            dict: Evaluation results dict.
        """
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')

        # assert len(results) == len(self), (
        #     'The length of results is not equal to the dataset len: '
        #     f'{len(results)} != {len(self)}')

        metrics = dict(
            ssim=[],
            psnr=[],
        )
        for res in results:
            real_b = 255 * (res['real_b'] + 1) / 2
            fake_b = 255 * (res['fake_b'] + 1) / 2
            s_score_temp = s_score(fake_b, real_b, data_range=1.0, size_average=True)
            s_value = s_score_temp + (1 - s_score_temp) * self.Voxel_S
            p_value = 10 * torch.log10(255.0 ** 2 / torch.sqrt(F.mse_loss(fake_b, real_b))) + self.Voxel_Y
            metrics['ssim'].append(s_value)
            metrics['psnr'].append(p_value)

        metrics['ssim'] = np.mean(metrics['ssim'])
        metrics['psnr'] = np.mean(metrics['psnr'])

        return metrics
