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
from model.unet import UNet
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
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--model', default='vgg16bn_unet', type=str)
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


yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True).cuda()


if train_cfg.model == 'vgg16bn_unet':
    generator = vgg16bn_unet().cuda()

elif train_cfg.model == 'UNet':
    generator = UNet(12).cuda()


generator.load_state_dict(torch.load('weights/target_vgg16bn_unet_avenue_MF_256_90000.pth')['net_g'])


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
        frame_1 = clips[:, 0:3, :, :]  # (n, 12, 256, 256) 
        frame_2 = clips[:, 3:6, :, :]  # (n, 12, 256, 256) 
        frame_3 = clips[:, 6:9, :, :]  # (n, 12, 256, 256) 
        frame_4 = clips[:, 9:12, :, :]  # (n, 12, 256, 256) 
        f_target = clips[:, 12:15, :, :]  # (n, 12, 256, 256) 
            
        # pop() the used frame index, this can't work in train_dataset.__getitem__ because of multiprocessing.
        for index in indice:
            train_dataset.all_seqs[index].pop()
            if len(train_dataset.all_seqs[index]) == 0:
                train_dataset.all_seqs[index] = list(range(len(train_dataset.videos[index]) - 4))
                random.shuffle(train_dataset.all_seqs[index])
        
        frame_1_crop = ((frame_1[0] + 1 ) / 2)[(2,1,0),...]
        frame_2_crop = ((frame_2[0] + 1 ) / 2)[(2,1,0),...]
        frame_3_crop = ((frame_3[0] + 1 ) / 2)[(2,1,0),...]
        frame_4_crop = ((frame_4[0] + 1 ) / 2)[(2,1,0),...]
        f_target_crop = ((f_target[0] + 1 ) / 2)[(2,1,0),...]
        img_1 = to_pil_image(frame_1_crop)
        img_2 = to_pil_image(frame_2_crop)
        img_3 = to_pil_image(frame_3_crop)
        img_4 = to_pil_image(frame_4_crop)
        img_t = to_pil_image(f_target_crop)
        # Select BB
        results = yolo_model(img_4)
        areas = results.xyxy[0]
        
        res_dat = results.pandas().xyxy
        new_areas = []
        for i, area in enumerate(areas):
            area = area.tolist()
            if area[5]==0:
                xmin = area[0]
                ymin = area[1]
                xmax = area[2]
                ymax = area[3]
                    
                n_x = 1.5
                n_y = 1.2
                xmin = xmin - (n_x-1)*(xmax-xmin)
                ymin = ymin - (n_y-1)*(ymax-ymin)
                xmax = xmax + (n_x-1)*(xmax-xmin)
                ymax = ymax + (n_y-1)*(ymax-ymin)
                x = xmax - xmin
                y = ymax - ymin
                if y > x:
                    dif = (y-x)/2
                    xmax += dif
                    xmin -= dif
                if y < x:
                    dif = (x-y)/2
                    ymax += dif
                    ymin -= dif
                    
                if(xmin < 0):
                    xmin = 0
                if(ymin < 0):
                    ymin = 0
                if(xmax > 256):
                    xmax = 256
                if(ymax > 256):
                    ymax = 256
                
                new_areas.append([xmin, ymin, xmax, ymax])
        
        if len(new_areas)!=0:
            tframe_1 = torch.Tensor([])
            tframe_2 = torch.Tensor([])
            tframe_3 = torch.Tensor([])
            tframe_4 = torch.Tensor([])
            tframe_t = torch.Tensor([])
            
            for i, area in enumerate(new_areas):
                crop_img_1 = (TT(img_1.crop(area).resize((input_size,input_size))).view([1,3,input_size,input_size])*2)-1
                crop_img_2 = (TT(img_2.crop(area).resize((input_size,input_size))).view([1,3,input_size,input_size])*2)-1
                crop_img_3 = (TT(img_3.crop(area).resize((input_size,input_size))).view([1,3,input_size,input_size])*2)-1
                crop_img_4 = (TT(img_4.crop(area).resize((input_size,input_size))).view([1,3,input_size,input_size])*2)-1
                crop_img_t = (TT(img_t.crop(area).resize((input_size,input_size))).view([1,3,input_size,input_size])*2)-1
                tframe_1 = torch.cat([tframe_1,crop_img_1],0)
                tframe_2 = torch.cat([tframe_2,crop_img_2],0)
                tframe_3 = torch.cat([tframe_3,crop_img_3],0)
                tframe_4 = torch.cat([tframe_4,crop_img_4],0)
                tframe_t = torch.cat([tframe_t,crop_img_t],0)

            # if step % 20 == 0:
            #     img_1.save(f'crop_imgs/tester.png')
            #     save_image(((tframe_1+1)/2),f'crop_imgs/tester_cat_11.png')
            #     save_image(((tframe_2+1)/2),f'crop_imgs/tester_cat_22.png')
            #     save_image(((tframe_3+1)/2),f'crop_imgs/tester_cat_33.png')
            #     save_image(((tframe_4+1)/2),f'crop_imgs/tester_cat_44.png')


            img_1.save(f'crop_imgs/tester.png')
            save_image(((tframe_1[0]+1)/2),f'crop_imgs/tester_mid_1.png')
            save_image(((tframe_2[1]+1)/2),f'crop_imgs/tester_mid_2.png')
            save_image(((tframe_3[2]+1)/2),f'crop_imgs/tester_mid_3.png')
            save_image(((tframe_4[3]+1)/2),f'crop_imgs/tester_mid_4.png')


            frame_1 = tframe_1.cuda()
            frame_2 = tframe_2.cuda()
            frame_3 = tframe_3.cuda()
            frame_4 = tframe_4.cuda()
            f_target = tframe_t.cuda()
            
            f_input = torch.cat([frame_1[0],frame_2[1], frame_3[2], frame_4[3]], 0)
            # f_input = torch.cat([frame_1,frame_2, frame_3, frame_4], 1)
            
            f_input = f_input.reshape(-1,12,256,256)
            print("f_input:",f_input.shape)

            FG_frame = generator(f_input)
            save_image(((FG_frame+1)/2),f'crop_imgs/tester_mid_res.png')
            print("GOOD")
            quit()