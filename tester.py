import os
from glob import glob
import cv2
import time
import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import argparse
import random
from torchvision.utils import save_image
import time
from utils import *
from losses import *
import Dataset
from model.unet import UNet
from model.pix2pix_networks import PixelDiscriminator
from config import update_config
from model.flownet2.models import FlowNet2SD
from model.vgg16_unet import *

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--model', default='vgg16bn_unet', type=str)
parser.add_argument('--dataset', default='avenue', type=str, help='The name of the dataset to train.')
parser.add_argument('--iters', default=180000, type=int, help='The total iteration number.')
parser.add_argument('--input_size', default=128, type=int, help='The img size.')

parser.add_argument('--resume', default=None, type=str,
                    help='The pre-trained model to resume training with, pass \'latest\' or the model name.')
parser.add_argument('--save_interval', default=15000, type=int, help='Save the model every [save_interval] iterations.')
parser.add_argument('--val_interval', default=30000, type=int,
                    help='Evaluate the model every [val_interval] iterations, pass -1 to disable.')

args = parser.parse_args()
train_cfg = update_config(args, mode='train')
train_cfg.print_cfg()

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
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    print('Generator and discriminator are going to be trained from scratch.\n')

flow_net = FlowNet2SD()
flow_net.load_state_dict(torch.load('models/flownet2/FlowNet2-SD.pth')['state_dict'])
flow_net.cuda().eval()

adversarial_loss = Adversarial_Loss().cuda()
discriminate_loss = Discriminate_Loss().cuda()
gradient_loss = Gradient_Loss(3).cuda()
flow_loss = Flow_Loss().cuda()
intensity_loss = Intensity_Loss().cuda()

train_dataset = Dataset.train_dataset(train_cfg)

# Remember to set drop_last=True, because we need to use 4 frames to predict one frame.
train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_cfg.batch_size,
                              shuffle=True, num_workers=4, drop_last=True)

writer = SummaryWriter(f'tensorboard_log/{train_cfg.dataset}_bs{train_cfg.batch_size}')
start_iter = int(train_cfg.resume.split('_')[-1].split('.')[0]) if train_cfg.resume else 0
training = True
generator = generator.train()
discriminator = discriminator.train()

step = start_iter
while training:
    for indice, clips, flow_strs in train_dataloader:
        input_frames = clips[:, 0:12, :, :].cuda()  # (n, 12, 256, 256)
        target_frame = clips[:, 12:15, :, :].cuda()  # (n, 3, 256, 256)
        input_last = input_frames[:, 9:12, :, :].cuda()  # use for flow_loss
        # pop() the used frame index, this can't work in train_dataset.__getitem__ because of multiprocessing.
        for index in indice:
            train_dataset.all_seqs[index].pop()
            if len(train_dataset.all_seqs[index]) == 0:
                train_dataset.all_seqs[index] = list(range(len(train_dataset.videos[index]) - 4))
                random.shuffle(train_dataset.all_seqs[index])
        G_frame = generator(input_frames)
        print(input_last.shape)

        input_last = input_last.unsqueeze(2)
        target_frame = target_frame.unsqueeze(2)
        G_frame = G_frame.unsqueeze(2)

        gt_flow_input = torch.cat([input_last, target_frame], 2)
        pred_flow_input = torch.cat([input_last, G_frame], 2)
        flow_gt = (flow_net(gt_flow_input * 255.) / 255.).detach()  # Input for flownet2sd is in (0, 255).
        flow_pred = (flow_net(pred_flow_input * 255.) / 255.).detach()

        print(input_last.shape)
        print(gt_flow_input.shape)

        # save_image(((flow_gt+1)/2),f'crop_imgs/tester_flow_11.png')
        quit()
        time.sleep(3)
