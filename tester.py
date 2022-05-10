import torch
from PIL import Image
from glob import glob
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import argparse
import random

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
from fid_score import *

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--batch_size', default=4, type=int)
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

train_dataset = Dataset.train_dataset(train_cfg)

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

step = start_iter
while training:
    for indice, clips in train_dataloader:
        frame_1 = clips[:, 0:3, :, :].cuda()  # (n, 12, 256, 256) 
        frame_2 = clips[:, 3:6, :, :].cuda()  # (n, 12, 256, 256) 
        frame_3 = clips[:, 6:9, :, :].cuda()  # (n, 12, 256, 256) 
        frame_4 = clips[:, 9:12, :, :].cuda()  # (n, 12, 256, 256) 
        f_target = clips[:, 12:15, :, :].cuda()  # (n, 12, 256, 256) 
        f_input = torch.cat([frame_1,frame_2, frame_3, frame_4], 1)
        # pop() the used frame index, this can't work in train_dataset.__getitem__ because of multiprocessing.
        for index in indice:
            train_dataset.all_seqs[index].pop()
            if len(train_dataset.all_seqs[index]) == 0:
                train_dataset.all_seqs[index] = list(range(len(train_dataset.videos[index]) - 4))
                random.shuffle(train_dataset.all_seqs[index])
        # Forward
        # FG_frame = generator(f_input)
        frames = ((frame_1 + 1 ) / 2)[(0,3,2,1),...]
        FG_save = frame_1[0]

        save_image(frames[0],f'crop_imgs/testers.png')

        img_0 = to_pil_image(frames[0])
        img_1 = to_pil_image(frames[1])
        img_2 = to_pil_image(frames[2])
        img_3 = to_pil_image(frames[3])
        
        img_li = [img_0, img_1, img_2, img_3]
        print(type(img_li[0]))
        results = yolo_model(img_li)
        print(type(img_li[0]))
        quit()
        areas = results.xyxy
        
        res_dat = results.pandas().xyxy
        # print(res_dat)

        for i, batches in enumerate(areas):
            for j, area in enumerate(batches):
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

                new_area = (xmin, ymin, xmax, ymax)
                print(type(img_li[i]))
                quit()

                crop_image = Image.fromarray(img_li[i]).crop(new_area)
                crop_image.save(f'crop_imgs/tester_{i}_{j}_15.png')
                Image.fromarray(img_li[i]).save(f'crop_imgs/tester_{i}_{j}.png')
                

        quit()