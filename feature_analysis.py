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

    load_psnr_mean = np.load('scores/psnr_resize_256_180000_mean.npy', allow_pickle=True)
    load_psnr_max = np.load('scores/psnr_resize_256_180000_max.npy', allow_pickle=True)
    load_psnr_min = np.load('scores/psnr_resize_256_180000_min.npy', allow_pickle=True)
    load_psnrs = np.load('scores/psnr_resize_256_180000.npy', allow_pickle=True)
    load_targets = np.load('scores/target_count.npy', allow_pickle=True)
    

    labels = np.array([], dtype=np.int8)

    psnrs_mean = np.array([[]], dtype=np.float32)
    psnrs_max = np.array([[]], dtype=np.float32)
    psnrs_min = np.array([[]], dtype=np.float32)
    psnrs_test = np.array([[]], dtype=np.float32)

    # len(load_psnr_mean) = 21 (dataset)
    for i in range(len(load_psnr_mean)):
        temp_mean = np.array([], dtype=np.float32)
        temp_max = np.array([], dtype=np.float32)
        temp_min = np.array([], dtype=np.float32)
        temp_test = np.array([], dtype=np.float32)

        for j in range(len(load_psnrs[i])):
            psnr_mean = torch.mean(load_psnrs[i][j])
            psnr_max = torch.max(load_psnrs[i][j])
            psnr_min = torch.min(load_psnrs[i][j])
            psnr_test = psnr_max - psnr_min

            temp_mean = np.append(temp_mean, psnr_mean)
            temp_max  = np.append(temp_max, psnr_max)
            temp_min  = np.append(temp_min, psnr_min)
            temp_test = np.append(temp_test, psnr_test)

    #     psnrs_mean = np.concatenate((psnrs_mean,[temp_mean]), axis=0)
    #     psnrs_max = np.concatenate( (psnrs_max ,[temp_max ]), axis=0)
    #     psnrs_min = np.concatenate( (psnrs_min ,[temp_min ]), axis=0)
    #     psnrs_test = np.concatenate((psnrs_test,[temp_test]), axis=0)


    print(psnrs_min)
    # quit()

    for i in range(21):
        labels = np.concatenate((labels, gt[i][4:]), axis=0)  # Exclude the first 4 unpredictable frames in gt.
        # print(len(gt[i][4:]))
    
    
    d_size = [1435, 1207, 919, 943, 1003, 1279, 601, 32, 1171, 837, 468, 1267, 545, 503, 997, 736, 422, 290, 244, 269, 72]

    # for j, d in enumerate(d_size):
    #     print(psnrs_mean[j])

    cur_point = 0
    cur_iter = 1
    for j, d in enumerate(d_size):
        # print(load_psnrs[j])

        # fig, ax1 = plt.subplots()
        # ax1.plot(range(d), load_psnr_min[j], color='red', label="min")
        # ax1.plot(range(d), load_psnr_mean[j], color='orange', label="mean")
        # ax1.plot(range(d), load_psnr_max[j], color='green', label="max")
        # # ax1.plot(range(d), psnr_test[j], color='black', label="max")
        
        # ax1.legend(['min', 'mean','max'], loc='center', bbox_to_anchor=(1.3,0.5))
        # ax1.tick_params(axis='y', labelcolor="red")

        # ax2 = ax1.twinx()
        
        # ax2.plot(range(d), labels[cur_point:cur_point+d], color="blue")
        # ax2.tick_params(axis='y', labelcolor="blue")
        # plt.savefig(f'scores/{cur_iter}_test_graph.png')

        # plt.clf()
        # print(load_psnr_min[j])
        fig, ax1 = plt.subplots()
        # ax1.plot(range(d), (load_psnr_min[j]), color='red', label="min")
        ax1.plot(range(d), load_psnr_mean[j], color='orange', label="mean")
        # ax1.plot(range(d), load_psnr_max[j], color='green', label="max")
        # ax1.plot(range(d), psnr_test[j], color='black', label="max")
        
        # ax1.legend(['min', 'mean','max'], loc='center', bbox_to_anchor=(1.3,0.5))
        # ax1.tick_params(axis='y', labelcolor="red")

        ax2 = ax1.twinx()
        
        ax2.plot(range(d), labels[cur_point:cur_point+d], color="blue")
        ax2.tick_params(axis='y', labelcolor="blue")
        # plt.savefig(f'scores/{cur_iter}_test_graph.png')
        plt.savefig(f'scores/{cur_iter}_mean_graph.png')

        plt.clf()

        cur_point += d
        cur_iter += 1

    fpr_min, tpr_min, thresholds = metrics.roc_curve(labels, psnrs_min, pos_label=0)
    fpr_max, tpr_max, thresholds = metrics.roc_curve(labels, psnrs_max, pos_label=0)
    fpr_mean, tpr_mean, thresholds = metrics.roc_curve(labels, psnrs_mean, pos_label=0)
    fpr_test, tpr_test, thresholds = metrics.roc_curve(labels, psnrs_test, pos_label=0)
    
    auc_min = metrics.auc(fpr_min, tpr_min)
    auc_mean = metrics.auc(fpr_mean, tpr_mean)
    auc_max = metrics.auc(fpr_max, tpr_max)
    auc_test = metrics.auc(fpr_test, tpr_test)
    

    print(f'AUC_max: {auc_max}\n')
    print(f'AUC_min: {auc_min}\n')
    print(f'AUC_mean: {auc_mean}\n')
    print(f'AUC_test: {auc_test}\n')
    

    return auc_min


if __name__ == '__main__':
    args = parser.parse_args()
    test_cfg = update_config(args, mode='test')
    test_cfg.print_cfg()
    val(test_cfg)
