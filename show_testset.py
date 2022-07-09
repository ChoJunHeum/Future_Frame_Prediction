from imghdr import tests
import numpy as np
import os
import time
import torch
import argparse
from PIL import Image
import io
from sklearn import metrics
import matplotlib.pyplot as plt
from model.vgg16_unet import *
from model.pix2pix_networks import PixelDiscriminator

from config import update_config
from Dataset import Label_loader
from util import psnr_error, psnr_error_target
import Dataset
from model.unet import UNet

from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import ToTensor

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--dataset', default='avenue', type=str, help='The name of the dataset to train.')
parser.add_argument('--trained_model', default=None, type=str, help='The pre-trained model to evaluate.')
parser.add_argument('--show_curve', action='store_true',
                    help='Show and save the psnr curve real-timely, this drops fps.')
parser.add_argument('--input_size', default=256, type=int, help='The img size.')


def val(cfg, model=None, dis_model=None, iters=None):

    # yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True).cuda()

    video_folders = os.listdir(cfg.test_data)
    video_folders.sort()
    video_folders = [os.path.join(cfg.test_data, aa) for aa in video_folders]
    TT = ToTensor()


    gt_loader = Label_loader(cfg, video_folders)  # Get gt labels.
    gt = gt_loader()

    with torch.no_grad():
        for i, folder in enumerate(video_folders):
            label = gt[i][4:]

            dataset = Dataset.test_dataset(cfg, folder)

            for j, clip in enumerate(dataset):
                if label[j] == 1:
                    target_np = clip[12:15, :, :]

                    f_target = torch.from_numpy(target_np).unsqueeze(0).cuda()
                    f_target_crop = ((f_target[0] + 1 ) / 2)[(2,1,0),...]   


                    save_image(f_target_crop,f'crop_imgs/ano/tester_{i}_{j+1}.png')

                    print(f'\rDetecting: {i} / {j + 1}\t Anomaly: {label[j]}', end='\n')

            # break



if __name__ == '__main__':
    args = parser.parse_args()
    test_cfg = update_config(args, mode='test')
    test_cfg.print_cfg()
    val(test_cfg)
    # Uncomment this to test the AUC mechanism.
    # labels = [0,  0,   0,   0,   0,  1,   1,    1,   0,  1,   0,    0]
    # scores = [0, 1/8, 2/8, 1/8, 1/8, 3/8, 6/8, 7/8, 5/8, 8/8, 2/8, 1/8]
    # fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    # print(fpr)
    # print('~~~~~~~~~~~~`')
    # print(tpr)
    # print('~~~~~~~~~~~~`')
    # print(thresholds)
    # print('~~~~~~~~~~~~`')