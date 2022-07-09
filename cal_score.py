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

    load_psnr_mean = np.load('scores/psnr_resize_256_test_mean.npy', allow_pickle=True)
    load_psnr_max = np.load('scores/psnr_resize_256_test_max.npy', allow_pickle=True)
    load_psnr_min = np.load('scores/psnr_resize_256_test_min.npy', allow_pickle=True)

    labels = np.array([], dtype=np.int8)
    d_scores = np.array([], dtype=np.float32)
    min_scores = np.array([], dtype=np.float32)
    max_scores = np.array([], dtype=np.float32)
    mean_scores = np.array([], dtype=np.float32)
    psnrs_mean = np.array([], dtype=np.float32)
    psnrs_max = np.array([], dtype=np.float32)
    psnrs_min = np.array([], dtype=np.float32)
    
    for i in range(len(load_psnr_mean)):

        # max_score = np.max(load_score[i])
        # min_score = np.min(load_score[i])
        # mean_score = np.mean(load_score[i])
        # d_score = min_score
        psnr_mean = load_psnr_mean[i]
        psnr_max = load_psnr_max[i]
        psnr_min = load_psnr_min[i]

        # print(min(psnr_mean))
        # quit()

        psnr_mean -= min(psnr_mean)  # distance = (distance - min) / (max - min)
        psnr_mean /= max(psnr_mean)
        
        psnr_min -= min(psnr_min)  # distance = (distance - min) / (max - min)
        psnr_min /= max(psnr_min)
        
        psnr_max -= min(psnr_max)  # distance = (distance - min) / (max - min)
        psnr_max /= max(psnr_max)

        # d_score -= np.min(d_score)  # distance = (distance - min) / (max - min)
        # d_score /= np.max(d_score)

        # d_scores = np.concatenate((d_scores, [d_score]), axis=0)
        # min_scores = np.concatenate((min_scores, [min_score]), axis=0)
        # max_scores = np.concatenate((max_scores, [max_score]), axis=0)
        # mean_scores = np.concatenate((mean_scores, [mean_score]), axis=0)
        
        # psnrs_mean = np.concatenate((psnrs_mean, psnr_mean), axis=0)
        # psnrs_max = np.concatenate((psnrs_max, psnr_max), axis=0)
        psnrs_min = np.concatenate((psnrs_min, psnr_min), axis=0)
        
    print(len(psnrs_min))

    for i in range(21):
        labels = np.concatenate((labels, gt[i][4:]), axis=0)  # Exclude the first 4 unpredictable frames in gt.
        # print(len(gt[i][4:]))

    # fpr_s, tpr_s, thresholds_s = metrics.roc_curve(labels, d_scores, pos_label=0)
    # # fpr_p, tpr_p, thresholds_s = metrics.roc_curve(labels, psnrs, pos_label=0)
    # fpr_mins, tpr_mins, thresholds_s = metrics.roc_curve(labels, min_scores, pos_label=0)
    # fpr_maxs, tpr_maxs, thresholds_s = metrics.roc_curve(labels, max_scores, pos_label=0)
    # fpr_means, tpr_means, thresholds_s = metrics.roc_curve(labels, mean_scores, pos_label=0)
    
    
    
    d_size = [1435, 1207, 919, 943, 1003, 1279, 601, 32, 1171, 837, 468, 1267, 545, 503, 997, 736, 422, 290, 244, 269, 72]

    cur_point = 0
    cur_iter = 1

    for j, d in enumerate(d_size):

        fig, ax1 = plt.subplots()
        ax1.plot(range(d), load_psnr_min[j], color='red')
        ax1.tick_params(axis='y', labelcolor="red")

        ax2 = ax1.twinx()
        
        ax2.plot(range(d), labels[cur_point:cur_point+d], color="blue")
        ax2.tick_params(axis='y', labelcolor="blue")
        plt.savefig(f'scores/{cur_iter}_test_graph.png')


        plt.clf()
    

        cur_point += d
        cur_iter += 1

    fpr_min, tpr_min, thresholds = metrics.roc_curve(labels, psnrs_min, pos_label=0)

    # auc_s = metrics.auc(fpr_s, tpr_s)
    # auc_p = metrics.auc(fpr_p, tpr_p)
    # auc_ma = metrics.auc(fpr_maxs, tpr_maxs)   
    auc_mi = metrics.auc(fpr_min, tpr_min)
    # auc_me = metrics.auc(fpr_means, tpr_means)
    

    # print(f'AUC_d: {auc_s}\n')
    # print(f'AUC_p: {auc_p}\n')
    # print(f'AUC_ma: {auc_ma}\n')
    print(f'AUC_mi: {auc_mi}\n')
    # print(f'AUC_me: {auc_me}\n')

    return auc_mi


if __name__ == '__main__':
    args = parser.parse_args()
    test_cfg = update_config(args, mode='test')
    test_cfg.print_cfg()
    val(test_cfg)
