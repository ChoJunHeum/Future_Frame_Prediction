import torch
from PIL import Image
from glob import glob
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import argparse
import random
import cv2

from curses import doupdate
import os
from glob import glob
import time
import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import argparse
import random

from util import *
from losses import *
import Dataset
from models.unet import UNet
from model.vgg16_unet import *
from model.pix2pix_networks import PixelDiscriminator
from model.flownet2.models import FlowNet2SD

# from models.liteFlownet import lite_flownet as lite_flow
from config import update_config
# from models.flownet2.models import FlowNet2SD
from evaluate_target import val
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import ToTensor
from fid_score import *

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--batch_size', default=5, type=int)
parser.add_argument('--model', default='UNet', type=str)
parser.add_argument('--dataset', default='avenue', type=str, help='The name of the dataset to train.')
parser.add_argument('--iters', default=180000, type=int, help='The total iteration number.')
parser.add_argument('--input_size', default=256, type=int, help='The img size.')
parser.add_argument('--trained_model', default=None, type=str, help='The pre-trained model to evaluate.')

parser.add_argument('--resume', default=None, type=str,
                    help='The pre-trained model to resume training with, pass \'latest\' or the model name.')
parser.add_argument('--save_interval', default=15000, type=int, help='Save the model every [save_interval] iterations.')
parser.add_argument('--val_interval', default=30000, type=int,
                    help='Evaluate the model every [val_interval] iterations, pass -1 to disable.')


args = parser.parse_args()
train_cfg = update_config(args, mode='train')
train_cfg.print_cfg()


generator = UNet(input_channels=12, output_channel=3).cuda()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=train_cfg.g_lr)

generator.load_state_dict(torch.load('weights/avenue_15000.pth')['net_g'])


train_dataset = Dataset.train_dataset(train_cfg)

# Remember to set drop_last=True, because we need to use 4 frames to predict one frame.
train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_cfg.batch_size,
                              shuffle=True, num_workers=4, drop_last=True)

input_size = train_cfg.input_size

writer = SummaryWriter(f'tensorboard_log/target_{train_cfg.model}_{train_cfg.dataset}_resize_{input_size}_{train_cfg.iters}')
start_iter = int(train_cfg.resume.split('_')[-1].split('.')[0]) if train_cfg.resume else 0
training = True
generator = generator.train()
fid_s=30
earlyFlag = False
data_name = train_cfg.dataset
TT = ToTensor()

step = start_iter
while training:
    for indice, clips, flow_strs in train_dataloader:
        frame_1 = clips[:, 0:3, :, :].cuda()  # (n, 12, 256, 256)
        frame_2 = clips[:, 3:6, :, :].cuda()  # (n, 12, 256, 256)
        frame_3 = clips[:, 6:9, :, :].cuda()  # (n, 12, 256, 256)
        frame_4 = clips[:, 9:12, :, :].cuda()  # (n, 12, 256, 256)
        f_target = clips[:, 12:15, :, :].cuda()  # (n, 12, 256, 256)
            
        # pop() the used frame index, this can't work in train_dataset.__getitem__ because of multiprocessing.
        for index in indice:
            train_dataset.all_seqs[index].pop()

            save_image(((frame_1[0]+1)/2),f'crop_imgs/tester_frame_1.png')
            save_image(((frame_2[1]+1)/2),f'crop_imgs/tester_frame_2.png')
            save_image(((frame_3[2]+1)/2),f'crop_imgs/tester_frame_3.png')
            save_image(((frame_4[3]+1)/2),f'crop_imgs/tester_frame_4.png')

            f_input = torch.cat([frame_1[0],frame_2[1], frame_3[2], frame_4[3]], 0)
            
            f_input = f_input.reshape(-1,12,256,256)
            print("f_input:",f_input.shape)

            FG_frame = generator(f_input)
            save_image(((FG_frame+1)/2),f'crop_imgs/tester_res.png')
            print("GOOD")
            quit()