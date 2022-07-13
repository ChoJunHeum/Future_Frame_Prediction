from imghdr import tests
import numpy as np
import os
import time
import torch
import argparse
import cv2
from PIL import Image
import io
from sklearn import metrics
import matplotlib.pyplot as plt
from model.vgg16_unet import *
from model.pix2pix_networks import PixelDiscriminator

from config import update_config
from Dataset import Label_loader
from util import psnr_error
import Dataset
from model.unet import UNet

from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--dataset', default='avenue', type=str, help='The name of the dataset to train.')
parser.add_argument('--trained_model', default=None, type=str, help='The pre-trained model to evaluate.')
parser.add_argument('--show_curve', action='store_true',
                    help='Show and save the psnr curve real-timely, this drops fps.')
parser.add_argument('--input_size', default=256, type=int, help='The img size.')


def val(cfg, model=None):

    video_folders = os.listdir(cfg.test_data)
    video_folders.sort()
    video_folders = [os.path.join(cfg.test_data, aa) for aa in video_folders]
    
    gt_loader = Label_loader(cfg, video_folders)  # Get gt labels.
    gt = gt_loader()
    for folder in range(1, 14):
        load_psnr = np.load(f'scores/psnrs_{folder}.npy', allow_pickle=True) # list of tensors
        load_psnr_min = np.load(f'scores/psnrs_min_{folder}.npy', allow_pickle=True) # list of tensors of shape[0]
        load_patch_size = np.load(f'scores/patch_size_{folder}.npy', allow_pickle=True) # list of tensors

        labels = np.array([], dtype=np.int8)
        min_sizes = []
        min_psnrs = []
        for i in range(len(load_psnr)):
            patch_size_ = load_patch_size[i]
            psnr_ = load_psnr[i].numpy()[:-1]

            arg_min_size_ = np.argmin(psnr_)
            min_size = patch_size_[arg_min_size_]
            min_sizes.append(min_size)
            min_psnrs.append(np.min(psnr_))

        s_max = np.max(min_sizes)
        weights = (np.ones_like(min_sizes) * s_max) / min_sizes
        # min_psnrs *= weights

        min_psnrs -= min(min_psnrs)  # distance = (distance - min) / (max - min)
        min_psnrs /= max(min_psnrs)
            
        # print(len(min_psnrs))

        # for i in range(21):
        #     labels = np.concatenate((labels, gt[i][4:]), axis=0)  # Exclude the first 4 unpredictable frames in gt.
        # labels = gt[0][4:]
            # print(len(gt[i][4:]))

        # fpr_s, tpr_s, thresholds_s = metrics.roc_curve(labels, d_scores, pos_label=0)
        # # fpr_p, tpr_p, thresholds_s = metrics.roc_curve(labels, psnrs, pos_label=0)
        # fpr_mins, tpr_mins, thresholds_s = metrics.roc_curve(labels, min_scores, pos_label=0)
        # fpr_maxs, tpr_maxs, thresholds_s = metrics.roc_curve(labels, max_scores, pos_label=0)
        # fpr_means, tpr_means, thresholds_s = metrics.roc_curve(labels, mean_scores, pos_label=0)
        
        
        # d_size = [1435]
        # d_size = [1435, 1207, 919, 943, 1003, 1279, 601, 32, 1171, 837, 468, 1267, 545, 503, 997, 736, 422, 290, 244, 269, 72]


        # fig, ax1 = plt.subplots()
        # ax1.plot(range(d_size[folder]), min_psnrs, color='red')
        # ax1.tick_params(axis='y', labelcolor="red")

        # ax2 = ax1.twinx()
        
        # ax2.plot(range(1435), labels[:1435], color="blue")
        # ax2.tick_params(axis='y', labelcolor="blue")
        # plt.savefig(f'scores/tmp_test_graph.png')
        # plt.clf()

        fpr_min, tpr_min, thresholds = metrics.roc_curve(gt[folder-1][4:], min_psnrs, pos_label=0)

        # auc_s = metrics.auc(fpr_s, tpr_s)
        # auc_p = metrics.auc(fpr_p, tpr_p)
        # auc_ma = metrics.auc(fpr_maxs, tpr_maxs)   
        auc_mi = metrics.auc(fpr_min, tpr_min)
        # auc_me = metrics.auc(fpr_means, tpr_means)
        

        # print(f'AUC_d: {auc_s}\n')
        # print(f'AUC_p: {auc_p}\n')
        # print(f'AUC_ma: {auc_ma}\n')
        print(f'{folder} \t AUC_mi: {auc_mi}\n')
    # print(f'AUC_me: {auc_me}\n')

    return auc_mi


if __name__ == '__main__':
    args = parser.parse_args()
    test_cfg = update_config(args, mode='test')
    test_cfg.print_cfg()
    val(test_cfg)
