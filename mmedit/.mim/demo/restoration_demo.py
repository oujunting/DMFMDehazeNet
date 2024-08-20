# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import mmcv
import torch

from mmedit.apis import init_model, restoration_inference
from mmedit.core import tensor2img
from mmedit.utils import modify_args


def parse_args():
    modify_args()
    parser = argparse.ArgumentParser(description='Restoration demo')
    parser.add_argument('--config',
                        default="/home/zhangzifan/MaintoCode/2023-4-03-01/sr_config/ablation_exp6_two_stage_asyconv_elan_ka_final_l1.py",
                        help='test config file path')
    parser.add_argument('--checkpoint',
                        default="/home/zhangzifan/MaintoCode/2023-4-03-01/work_dirs/face_nonblind/ablation_exp6_two_stage_asyconv_elan_ka_final_l1/iter_10000.pth",
                        help='checkpoint file')
    parser.add_argument('--img_path_dir', default="/home/zhangzifan/MaintoCode/2023-4-03-01/DIV2K/test",
                        help='path to input image file')
    parser.add_argument('--save_path_dir', default="/home/zhangzifan/MaintoCode/2023-4-03-01/DIV2K/pred",
                        help='path to save generation result')
    parser.add_argument(
        '--imshow', action='store_true', help='whether show image with opencv')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument(
        '--ref-path', default=None, help='path to reference image file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.device < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)

    model = init_model(args.config, args.checkpoint, device=device)

    classes_list = os.listdir(args.img_path_dir)
    for classes in classes_list:
        file_list = os.listdir(os.path.join(args.img_path_dir, classes))
        if not os.path.exists(os.path.join(args.img_path_dir, classes)):
            os.mkdir(os.path.join(args.save_path_dir, classes))
        for file in file_list:
            img_path = os.path.join(args.img_path_dir, classes, file)
            if args.ref_path:  # Ref-SR
                output = restoration_inference(model, img_path, args.ref_path)
            else:  # SISR
                output = restoration_inference(model, img_path)
            output = tensor2img(output)
            mmcv.imwrite(output, os.path.join(args.save_path_dir, classes, file))
            if args.imshow:
                mmcv.imshow(output, 'predicted restoration result')


if __name__ == '__main__':
    main()
