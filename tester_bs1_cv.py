import torch
from PIL import Image
from glob import glob
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import argparse
import random
import cv2

from util import *
from losses import *
import Dataset
from model.vgg16_unet import *
from model.pix2pix_networks import PixelDiscriminator

# from models.liteFlownet import lite_flownet as lite_flow
from config import update_config
# from models.flownet2.models import FlowNet2SD
from evaluate_ft import val
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import ToTensor
from fid_score import *

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--model', default='vgg16bn_unet', type=str)
parser.add_argument('--dataset', default='avenue', type=str, help='The name of the dataset to train.')
parser.add_argument('--iters', default=60000, type=int, help='The total iteration number.')
parser.add_argument('--resume', default=None, type=str,
                    help='The pre-trained model to resume training with, pass \'latest\' or the model name.')
parser.add_argument('--save_interval', default=1000, type=int, help='Save the model every [save_interval] iterations.')
parser.add_argument('--val_interval', default=1000, type=int,
                    help='Evaluate the model every [val_interval] iterations, pass -1 to disable.')


args = parser.parse_args()
train_cfg = update_config(args, mode='train')
train_cfg.print_cfg()


yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True).cuda()

generator = vgg16bn_unet().cuda()
discriminator = PixelDiscriminator(input_nc=3).cuda()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=train_cfg.g_lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=train_cfg.d_lr)

if train_cfg.resume:
    generator.load_state_dict(torch.load(train_cfg.resume)['net_g'])
    discriminator.load_state_dict(torch.load(train_cfg.resume)['net_d'])
    optimizer_G.load_state_dict(torch.load(train_cfg.resume)['optimizer_g'])
    optimizer_D.load_state_dict(torch.load(train_cfg.resume)['optimizer_d'])
    print(f'Pre-trained generator and discriminator have been loaded.\n')
else:
    # generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    print('Generator and discriminator are going to be trained from scratch.\n')

adversarial_loss = Adversarial_Loss().cuda()
discriminate_loss = Discriminate_Loss().cuda()
gradient_loss = Gradient_Loss(3).cuda()
intensity_loss = Intensity_Loss().cuda()

train_dataset = Dataset.train_target_dataset(train_cfg)

# Remember to set drop_last=True, because we need to use 4 frames to predict one frame.
train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_cfg.batch_size,
                              shuffle=True, num_workers=4, drop_last=True)

writer = SummaryWriter(f'tensorboard_log/{train_cfg.model}_{train_cfg.dataset}_bs{train_cfg.batch_size}')
start_iter = int(train_cfg.resume.split('_')[-1].split('.')[0]) if train_cfg.resume else 0
training = True
generator = generator.train()
discriminator = discriminator.train()
fid_s=30
earlyFlag = False

data_name = args.dataset
TT = ToTensor()

step = start_iter
while training:
    for indice, clips in train_dataloader:

        # clip: 5 * [cv2 tensor]
        # cv2 tensor: 
        
        # pop() the used frame index, this can't work in train_dataset.__getitem__ because of multiprocessing.
        for index in indice:
            train_dataset.all_seqs[index].pop()
            if len(train_dataset.all_seqs[index]) == 0:
                train_dataset.all_seqs[index] = list(range(len(train_dataset.videos[index]) - 4))
                random.shuffle(train_dataset.all_seqs[index])
        # Forward
        # FG_frame = generator(f_input)

        frame_1 = clips[0][:,:,:,[2,1,0]]
        frame_2 = clips[1][:,:,:,[2,1,0]]
        frame_3 = clips[2][:,:,:,[2,1,0]]
        frame_4 = clips[3][:,:,:,[2,1,0]]
        f_target = clips[4][:,:,:,[2,1,0]]

        print(frame_1.shape)
        temp = frame_1.view([-1, 3,640, 360])
        print(temp.shape)

        results = yolo_model(temp)

        print(results)
        quit()
        
        areas = results.xyxy[0]
        
        res_dat = results.pandas().xyxy
        cv2.imwrite(f'crop_imgs/tester.png', frame_1)
        for i, area in enumerate(areas):
            area = area.tolist()

            xmin = area[0]
            ymin = area[1]
            xmax = area[2]
            ymax = area[3]
                
            n_x = 2
            n_y = 1.5

            xmin = xmin - (n_x-1)*(xmax-xmin)
            ymin = ymin - (n_y-1)*(ymax-ymin)
            xmax = xmax + (n_x-1)*(xmax-xmin)
            ymax = ymax + (n_y-1)*(ymax-ymin)

            new_area = [xmin, ymin, xmax, ymax]
            crop_image = frame_1[xmin, ymin, xmax, ymax]

            cv2.imwrite(f'crop_imgs/tester_{i}.png', crop_image)
        
        quit()

        tframe_1 = torch.Tensor([])
        tframe_2 = torch.Tensor([])
        tframe_3 = torch.Tensor([])
        tframe_4 = torch.Tensor([])
        tframe_t = torch.Tensor([])
        

        for area in new_area:
            crop_img_1 = TT(img_1.crop(new_area))
            crop_img_2 = TT(img_2.crop(new_area))
            crop_img_3 = TT(img_3.crop(new_area))
            crop_img_4 = TT(img_4.crop(new_area))
            crop_img_t = TT(img_t.crop(new_area))

            print(crop_img_1.shape)
            print(crop_img_1.view([-1,3,256,256]).shape)

            torch.cat([tframe_1,crop_img_1],0)
            torch.cat([tframe_2,crop_img_2],0)
            torch.cat([tframe_3,crop_img_3],0)
            torch.cat([tframe_4,crop_img_4],0)
            torch.cat([tframe_t,crop_img_5],0)
            
            # target_data[0].append(crop_img_1)
            # target_data[1].append(crop_img_2)
            # target_data[2].append(crop_img_3)
            # target_data[3].append(crop_img_4)
            # target_data[4].append(crop_img_t)
            
        # train_X = torch.tensor(target_data)

        print(tframe_1.shape)

        quit()


        crop_image = img.crop(new_area)

        crop_image.save(f'crop_imgs/tester_{i}.png')
        



        quit()