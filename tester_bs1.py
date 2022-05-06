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
parser.add_argument('--save_interval', default=10000, type=int, help='Save the model every [save_interval] iterations.')
parser.add_argument('--val_interval', default=10000, type=int,
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
TT = ToTensor()

try:
    step = start_iter
    while training:
        for indice, clips in train_dataloader:
            frame_1 = clips[:, 0:3, :, :].cuda()  # (n, 12, 256, 256) 
            frame_2 = clips[:, 3:6, :, :].cuda()  # (n, 12, 256, 256) 
            frame_3 = clips[:, 6:9, :, :].cuda()  # (n, 12, 256, 256) 
            frame_4 = clips[:, 9:12, :, :].cuda()  # (n, 12, 256, 256) 
            f_target = clips[:, 12:15, :, :].cuda()  # (n, 12, 256, 256) 
                
            # pop() the used frame index, this can't work in train_dataset.__getitem__ because of multiprocessing.
            for index in indice:
                train_dataset.all_seqs[index].pop()
                if len(train_dataset.all_seqs[index]) == 0:
                    train_dataset.all_seqs[index] = list(range(len(train_dataset.videos[index]) - 4))
                    random.shuffle(train_dataset.all_seqs[index])
            
            frame_1 = ((frame_1[0] + 1 ) / 2)[(2,1,0),...]
            frame_2 = ((frame_2[0] + 1 ) / 2)[(2,1,0),...]
            frame_3 = ((frame_3[0] + 1 ) / 2)[(2,1,0),...]
            frame_4 = ((frame_4[0] + 1 ) / 2)[(2,1,0),...]
            f_target = ((f_target[0] + 1 ) / 2)[(2,1,0),...]

            img_1 = to_pil_image(frame_1)
            img_2 = to_pil_image(frame_2)
            img_3 = to_pil_image(frame_3)
            img_4 = to_pil_image(frame_4)
            img_t = to_pil_image(f_target)

            results = yolo_model(img_1)
            areas = results.xyxy[0]
            # print(areas)
            
            res_dat = results.pandas().xyxy

            new_areas = []

            for i, area in enumerate(areas):
                area = area.tolist()

                if area[4] > 0.6:

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
                    crop_img_1 = (TT(img_1.crop(area).resize((128,128))).view([1,3,128,128])*2)-1
                    crop_img_2 = (TT(img_2.crop(area).resize((128,128))).view([1,3,128,128])*2)-1
                    crop_img_3 = (TT(img_3.crop(area).resize((128,128))).view([1,3,128,128])*2)-1
                    crop_img_4 = (TT(img_4.crop(area).resize((128,128))).view([1,3,128,128])*2)-1
                    crop_img_t = (TT(img_t.crop(area).resize((128,128))).view([1,3,128,128])*2)-1

                    
                    # save_image(crop_img_1,f'crop_imgs/tester_1_{i}.png')
                    # save_image(crop_img_2,f'crop_imgs/tester_2_{i}.png')
                    # save_image(crop_img_3,f'crop_imgs/tester_3_{i}.png')
                    # save_image(crop_img_4,f'crop_imgs/tester_4_{i}.png')
                    # save_image(crop_img_t,f'crop_imgs/tester_t_{i}.png')


                    tframe_1 = torch.cat([tframe_1,crop_img_1],0)
                    tframe_2 = torch.cat([tframe_2,crop_img_2],0)
                    tframe_3 = torch.cat([tframe_3,crop_img_3],0)
                    tframe_4 = torch.cat([tframe_4,crop_img_4],0)
                    tframe_t = torch.cat([tframe_t,crop_img_t],0)
                
                # for i, frames in enumerate(tframe_1):
                #     save_image(frames,f'crop_imgs/tester_1_{i}.png')
                # for i, frames in enumerate(tframe_2):
                #     save_image(frames,f'crop_imgs/tester_2_{i}.png')
                # for i, frames in enumerate(tframe_3):
                #     save_image(frames,f'crop_imgs/tester_3_{i}.png')
                # for i, frames in enumerate(tframe_4):
                #     save_image(frames,f'crop_imgs/tester_4_{i}.png')
                # for i, frames in enumerate(tframe_t):
                #     save_image(frames,f'crop_imgs/tester_t_{i}.png')
                if step % 20 == 0:
                    img_1.save(f'crop_imgs/tester.png')
                    save_image(((tframe_1+1)/2),f'crop_imgs/tester_cat_11.png')
                    # print(areas)
                    # save_image(((tframe_2+1)/2),f'crop_imgs/tester_cat_22.png')
                    # save_image(((tframe_3+1)/2),f'crop_imgs/tester_cat_33.png')
                    # save_image(((tframe_4+1)/2),f'crop_imgs/tester_cat_44.png')

                bs_size = len(tframe_1)
                frame_1 = tframe_1.cuda()
                frame_2 = tframe_2.cuda()
                frame_3 = tframe_3.cuda()
                frame_4 = tframe_4.cuda()
                f_target = tframe_t.cuda()


                f_input = torch.cat([frame_1,frame_2, frame_3, frame_4], 1)
                # print(torch.min(f_input[0])) # torch.Size([3, 12, 256, 256])
                # print(torch.max(f_input[0])) # torch.Size([3, 12, 256, 256])
                # print(f_input.shape) # torch.Size([3, 12, 256, 256])
                # print(f_input[0].shape) # torch.Size([3, 12, 256, 256])
                
                
                FG_frame = generator(f_input)

                inte_fl = intensity_loss(FG_frame, f_target)
                grad_fl = gradient_loss(FG_frame, f_target)
                d_f_out, d_f_score = discriminator(FG_frame)
                g_fl = adversarial_loss(d_f_out)
                G_fl_t = 1. * inte_fl + 1. * grad_fl + 0.05 * g_fl

                # When training discriminator, don't train generator, so use .detach() to cut off gradients.
                d_ft, d_ft_s = discriminator(f_target)
                d_f_out_d, d_f_score_d = discriminator(FG_frame.detach())
                D_fl = discriminate_loss(d_ft, d_f_out_d)
                D_fl_s = discriminate_loss(d_ft_s, d_f_score_d)
                
                _, predicted = torch.max(d_f_score, 1)

                # Backward
                b_input = torch.cat([FG_frame.detach(), frame_4, frame_3, frame_2], 1)
                b_target = frame_1

                BG_frame = generator(b_input)

                inte_bl = intensity_loss(BG_frame, b_target)
                grad_bl = gradient_loss(BG_frame, b_target)
                d_b_out, d_b_score = discriminator(BG_frame) 
                g_bl = adversarial_loss(d_b_out)
                G_bl_t = 1. * inte_bl + 1. * grad_bl + 0.05 * g_bl

                # When training discriminator, don't train generator, so use .detach() to cut off gradients.
                d_b_t, d_bt_s = discriminator(b_target)
                d_b_out_d, d_b_score_d = discriminator(BG_frame.detach())
                D_bl = discriminate_loss(d_b_t, d_b_out_d)
                D_bl_s = discriminate_loss(d_bt_s, d_b_score_d)

                # Total Loss
                inte_l = inte_fl + inte_bl
                grad_l = grad_fl + grad_bl

                g_l = g_fl + g_bl
                G_l_t = G_fl_t + G_bl_t

                D_l = D_fl + D_bl + D_fl_s + D_bl_s

                # Or just do .step() after all the gradients have been computed, like the following way:
                optimizer_G.zero_grad()
                G_l_t.backward()
                optimizer_G.step()
                
                optimizer_D.zero_grad()
                D_l.backward()
                optimizer_D.step()


                torch.cuda.synchronize()
                time_end = time.time()
                if step > start_iter:  # This doesn't include the testing time during training.
                    iter_t = time_end - temp
                temp = time_end

                temp_FG_frame = ((FG_frame[0] + 1 ) / 2)
                temp_FT_frame = ((f_target[0] + 1 ) / 2)

                temp_BG_frame = ((BG_frame[0] + 1 ) / 2)
                temp_BT_frame = ((b_target[0] + 1 ) / 2)
                

                if step != start_iter:
                    if step % 20 == 0:
                        time_remain = (train_cfg.iters - step) * iter_t
                        eta = str(datetime.timedelta(seconds=time_remain)).split('.')[0]

                        # print(FG_frame.shape, f_target.shape)
                        # print(BG_frame.shape, b_target.shape)

                        # print("FG: ",FG_frame.shape)
                        # print("FT: " ,f_target.shape)
                        # print("BG: ",BG_frame.shape)
                        # print("BT: " ,b_target.shape)
                        
                        f_psnr = psnr_error(FG_frame, f_target)
                        b_psnr = psnr_error(BG_frame, b_target)

                        psnr = (f_psnr + b_psnr)/2

                        lr_g = optimizer_G.param_groups[0]['lr']
                        lr_d = optimizer_D.param_groups[0]['lr']


                        print(f"[{step}]  grad_fl: {grad_fl:.3f} | grad_bl: {grad_bl:.3f} | g_fl: {g_fl:.3f} | g_bl: {g_bl:.3f} "
                            f"| G_fl_total: {G_fl_t:.3f} | G_bl_total: {G_bl_t:.3f} | D_fl: {D_fl:.3f} | D_bl: {D_bl:.3f} | D_fl_s: {D_fl_s:.3f} | D_bl_s: {D_bl_s:.3f} | "
                            f"| f_psnr: {f_psnr:.3f} | b_psnr: {b_psnr:.3f} | ETA: {eta} | iter: {iter_t:.3f}s")

                        # save_FG_frame = ((FG_frame[0] + 1) / 2)
                        # save_FG_frame = save_FG_frame.cpu().detach()[(2, 1, 0), ...]
                        # save_F_target = ((f_target[0] + 1) / 2)
                        # save_F_target = save_F_target.cpu().detach()[(2, 1, 0), ...]

                        # save_BG_frame = ((BG_frame[0] + 1) / 2)
                        # save_BG_frame = save_BG_frame.cpu().detach()[(2, 1, 0), ...]
                        # save_B_target = ((b_target[0] + 1) / 2)
                        # save_B_target = save_B_target.cpu().detach()[(2, 1, 0), ...]

                        writer.add_scalar('psnr/forward/train_psnr', f_psnr, global_step=step)
                        writer.add_scalar('total_loss/forward/g_loss_total', G_fl_t, global_step=step)
                        writer.add_scalar('total_loss/forward/d_loss', D_fl, global_step=step)
                        writer.add_scalar('G_loss_total/forward/g_loss', g_fl, global_step=step)

                        writer.add_scalar('G_loss_total/forward/inte_loss', inte_fl, global_step=step)
                        writer.add_scalar('G_loss_total/forward/grad_loss', grad_fl, global_step=step)

                        writer.add_scalar('psnr/backward/train_psnr', b_psnr, global_step=step)
                        writer.add_scalar('total_loss/backward/g_loss_total', G_bl_t, global_step=step)
                        writer.add_scalar('total_loss/backward/d_loss', D_bl, global_step=step)
                        writer.add_scalar('G_loss_total/backward/g_loss', g_bl, global_step=step)

                        writer.add_scalar('G_loss_total/backward/inte_loss', inte_bl, global_step=step)
                        writer.add_scalar('G_loss_total/backward/grad_loss', grad_bl, global_step=step)

                    # if step % 1000 == 0:
                    #     save_image(save_FG_frame, f'training_imgs/{data_name}/{step}_FG_frame.png')
                    #     save_image(save_F_target, f'training_imgs/{data_name}/{step}_FT_frame_.png')
                        
                    #     save_image(save_BG_frame, f'training_imgs/{data_name}/{step}_BG_frame_.png')
                    #     save_image(save_B_target, f'training_imgs/{data_name}/{step}_BT_frame_.png')

                    # if step % int(train_cfg.iters / 100) == 0:
                    #     writer.add_image('image/FG_frame', save_FG_frame, global_step=step)
                    #     writer.add_image('image/f_target', save_F_target, global_step=step)
                    #     print()

                    #     writer.add_image('image/BG_frame', save_BG_frame, global_step=step)
                    #     writer.add_image('image/b_target', save_B_target, global_step=step)


                    if step % train_cfg.save_interval == 0:
                        model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                                    'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict()}
                        torch.save(model_dict, f'weights/target_{train_cfg.model}_{train_cfg.dataset}_resize_{step}.pth')
                        print(f'\nAlready saved: \'target_{train_cfg.model}_{train_cfg.dataset}_resize_{step}.pth\'.')

                    if step % train_cfg.val_interval == 0:
                        val_psnr = val(train_cfg, model=generator)
                        print("Val Score: ",val_psnr)
                        writer.add_scalar('results/val_psnr', val_psnr, global_step=step)
                        generator.train()
                    

            step += 1
            if step > train_cfg.iters:
                training = False
                model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                            'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict()}
                torch.save(model_dict, f'weights/latest_target_{train_cfg.dataset}_{step}.pth')
                break
except KeyboardInterrupt:
    print(f'\nStop early, model saved: \'latest_{train_cfg.dataset}_{step}.pth\'.\n')

    if glob(f'weights/latest*'):
        os.remove(glob(f'weights/latest*')[0])

    model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                  'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict()}
    torch.save(model_dict, f'weights/latest_{train_cfg.dataset}_{step}.pth')
