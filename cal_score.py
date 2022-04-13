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


def val(cfg, model=None):

    video_folders = os.listdir(cfg.test_data)
    video_folders.sort()
    video_folders = [os.path.join(cfg.test_data, aa) for aa in video_folders]
    
    gt_loader = Label_loader(cfg, video_folders)  # Get gt labels.
    gt = gt_loader()

    load_score = np.load('scores/score_discriminator.npy', allow_pickle=True)
    load_psnr = np.load('scores/psnr.npy', allow_pickle=True)
    
    # print(load_score)
    # np.savetxt('scores/save_1.txt', load_score, fmt='%s')

    # load_score = np.loadtxt("scores/save_1.txt", dtype=float)

    labels = np.array([], dtype=np.int8)
    d_scores = np.array([], dtype=np.float32)
    min_scores = np.array([], dtype=np.float32)
    max_scores = np.array([], dtype=np.float32)
    psnrs = np.array([], dtype=np.float32)

    print(load_psnr)
    for i in range(len(load_score)):

        max_score = np.max(load_score[i])
        min_score = np.min(load_score[i])

        d_score = min_score
        psnr = load_psnr[i]

        # d_score -= np.min(d_score)  # distance = (distance - min) / (max - min)
        # d_score /= np.max(d_score)

        d_scores = np.concatenate((d_scores, [d_score]), axis=0)
        min_scores = np.concatenate((min_scores, [min_score]), axis=0)
        max_scores = np.concatenate((max_scores, [max_score]), axis=0)
        psnrs = np.concatenate((psnrs, [psnr]), axis=0)


    for i in range(21):
        labels = np.concatenate((labels, gt[i][4:]), axis=0)  # Exclude the first 4 unpredictable frames in gt.
        print(len(gt[i][4:]))

    fpr_s, tpr_s, thresholds_s = metrics.roc_curve(labels, d_scores, pos_label=0)
    fpr_p, tpr_p, thresholds_s = metrics.roc_curve(labels, psnrs, pos_label=0)
    
    d_size = [1435, 1207, 919, 943, 1003, 1279, 601, 32, 1171, 837, 468, 1267, 545, 503, 997, 736, 422, 290, 244, 269, 72]

    cur_point = 0
    cur_iter = 1

    for d in d_size:

        plt.plot(range(d), max_scores[cur_point:cur_point+d])
        plt.savefig(f'scores/{cur_iter}_max_graph.png')

        plt.clf()

        plt.plot(range(d), min_scores[cur_point:cur_point+d])
        plt.savefig(f'scores/{cur_iter}_min_graph.png')

        plt.clf()

        plt.plot(range(d), labels[cur_point:cur_point+d])
        plt.savefig(f'scores/{cur_iter}_labels_graph.png')

        plt.clf()

        plt.plot(range(d), psnrs[cur_point:cur_point+d])
        plt.savefig(f'scores/{cur_iter}_psnr_graph.png')

        plt.clf()
    

        cur_point += d
        cur_iter += 1


    auc_s = metrics.auc(fpr_s, tpr_s)
    auc_p = metrics.auc(fpr_p, tpr_p)

    print(f'AUC_d: {auc_s}\n')
    print(f'AUC_p: {auc_p}\n')

    return auc_s


if __name__ == '__main__':
    args = parser.parse_args()
    test_cfg = update_config(args, mode='test')
    test_cfg.print_cfg()
    val(test_cfg)
