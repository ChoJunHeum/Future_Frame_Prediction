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

from ft_config import update_config
from Dataset import Label_loader
from utils import psnr_error
import Dataset
from models.unet import UNet

from torchvision.utils import save_image
from fid_score import *

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--dataset', default='avenue', type=str, help='The name of the dataset to train.')
parser.add_argument('--trained_model', default=None, type=str, help='The pre-trained model to evaluate.')
parser.add_argument('--show_curve', action='store_true',
                    help='Show and save the psnr curve real-timely, this drops fps.')
parser.add_argument('--fid', default=False, type=bool, help='Check FID Score')

def val(cfg, model=None):
    if model:  # This is for testing during training.
        generator = model
        generator.eval()
    else:
        generator = UNet(input_channels=12, output_channel=3).cuda().eval()
        generator.load_state_dict(torch.load('weights/' + cfg.trained_model)['net_g'])
        print(f'The pre-trained generator has been loaded from \'weights/{cfg.trained_model}\'.\n')

    video_folders = os.listdir(cfg.test_data)
    video_folders.sort()
    video_folders = [os.path.join(cfg.test_data, aa) for aa in video_folders]

    fps = 0
    psnr_group = []

    dataset_name = cfg.dataset

    with torch.no_grad():
        for i, folder in enumerate(video_folders):
            dataset = Dataset.test_dataset(cfg, folder)

            if not os.path.exists(f"results/{dataset_name}/f{i+1}"):
                os.makedirs(f"results/{dataset_name}/f{i+1}")

            psnrs = []
            save_num = 0

            for j, clip in enumerate(dataset):
                input_np = clip[0:12, :, :]
                target_np = clip[12:15, :, :]
                input_frames = torch.from_numpy(input_np).unsqueeze(0).cuda()
                target_frame = torch.from_numpy(target_np).unsqueeze(0).cuda()

                G_frame = generator(input_frames)
                test_psnr = psnr_error(G_frame, target_frame).cpu().detach().numpy()
                psnrs.append(float(test_psnr))
                res_temp = ((G_frame[0] + 1 ) / 2)[(2,1,0),...]
                save_image(res_temp, f'results/{dataset_name}/f{i+1}/{save_num}_img.jpg')
                save_num=save_num+1

                fid_num = 0

                # for g, t in zip(G_frame, target_frame):
                #     save_image(g, f'fid_img/avenue/gen/{fid_num}_gen_img.jpg')
                #     save_image(t, f'fid_img/avenue/tar/{fid_num}_tar_img.jpg')

                #     fid_num=fid_num+1

                #     fid = calculate_fid_given_paths(
                #         paths=['fid_img/avenue/gen', 'fid_img/avenue/tar'],
                #         batch_size=1,
                #         device='cuda',
                #         dims=2048
                #     )
                #     print("FID Score: ", fid)

                torch.cuda.synchronize()
                end = time.time()
                if j > 1:  # Compute fps by calculating the time used in one completed iteration, this is more accurate.
                    fps = 1 / (end - temp)
                temp = end
                print(f'\rDetecting: [{i + 1:02d}] {j + 1}/{len(dataset)}, {fps:.2f} fps, PSNR: {sum(psnrs)/len(psnrs):.2f}', end='')

            psnr_group.append(np.array(psnrs))
            print(psnr_group)

    print(torch.mean(psnr_group))
    return auc


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
