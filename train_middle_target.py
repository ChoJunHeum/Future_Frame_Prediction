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

parser.add_argument('--resume', default=None, type=str,
                    help='The pre-trained model to resume training with, pass \'latest\' or the model name.')
parser.add_argument('--save_interval', default=15000, type=int, help='Save the model every [save_interval] iterations.')
parser.add_argument('--val_interval', default=100000, type=int,
                    help='Evaluate the model every [val_interval] iterations, pass -1 to disable.')


args = parser.parse_args()
train_cfg = update_config(args, mode='train')
train_cfg.print_cfg()


yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True).cuda()


if train_cfg.model == 'vgg16bn_unet':
    generator = vgg16bn_unet().cuda()

elif train_cfg.model == 'UNet':
    generator = UNet(12).cuda()


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

adversarial_loss = Adversarial_Loss().cuda()
discriminate_loss = Discriminate_Loss().cuda()
gradient_loss = Gradient_Loss(3).cuda()
intensity_loss = Intensity_Loss().cuda()
flow_loss = Flow_Loss().cuda()

train_dataset = Dataset.train_dataset(train_cfg)

flow_net = FlowNet2SD()
flow_net.load_state_dict(torch.load('models/flownet2/FlowNet2-SD.pth')['state_dict'])
flow_net.cuda().eval()

# Remember to set drop_last=True, because we need to use 4 frames to predict one frame.
train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_cfg.batch_size,
                              shuffle=True, num_workers=4, drop_last=True)

input_size = train_cfg.input_size

writer = SummaryWriter(f'tensorboard_log/target_{train_cfg.model}_{train_cfg.dataset}_resize_{input_size}_{train_cfg.iters}')
start_iter = int(train_cfg.resume.split('_')[-1].split('.')[0]) if train_cfg.resume else 0
training = True
generator = generator.train()
discriminator = discriminator.train()
fid_s=30
earlyFlag = False
data_name = train_cfg.dataset
TT = ToTensor()

psnr_saves = np.array([], dtype=np.float32)

print(f'Train Info: target_{train_cfg.model}_{train_cfg.dataset}_resize_{input_size}_{train_cfg.iters}.pth.')


try:
    step = start_iter
    while training:
        for indice, clips, flow_strs in train_dataloader:
            frame_1 = clips[:, 0:3, :, :]  # (n, 12, 256, 256) 
            frame_2 = clips[:, 3:6, :, :]  # (n, 12, 256, 256) 
            f_target = clips[:, 6:9, :, :]  # (n, 12, 256, 256) 
            frame_4 = clips[:, 9:12, :, :]  # (n, 12, 256, 256) 
            frame_5 = clips[:, 12:15, :, :]  # (n, 12, 256, 256) 
                
            # pop() the used frame index, this can't work in train_dataset.__getitem__ because of multiprocessing.
            for index in indice:
                train_dataset.all_seqs[index].pop()
                if len(train_dataset.all_seqs[index]) == 0:
                    train_dataset.all_seqs[index] = list(range(len(train_dataset.videos[index]) - 4))
                    random.shuffle(train_dataset.all_seqs[index])
            
            frame_1_crop = ((frame_1[0] + 1 ) / 2)[(2,1,0),...]
            frame_2_crop = ((frame_2[0] + 1 ) / 2)[(2,1,0),...]
            frame_4_crop = ((frame_4[0] + 1 ) / 2)[(2,1,0),...]
            frame_5_crop = ((frame_5[0] + 1 ) / 2)[(2,1,0),...]
            f_target_crop = ((f_target[0] + 1 ) / 2)[(2,1,0),...]

            img_1 = to_pil_image(frame_1_crop)
            img_2 = to_pil_image(frame_2_crop)
            img_4 = to_pil_image(frame_4_crop)
            img_5 = to_pil_image(frame_5_crop)
            img_t = to_pil_image(f_target_crop)

            # Select BB
            results = yolo_model(img_2)
            areas = results.xyxy[0]
            
            res_dat = results.pandas().xyxy

            new_areas = []

            for i, area in enumerate(areas):
                area = area.tolist()

                if area[4] > 0.4 and area[5]==0:

                    xmin_ = area[0]
                    ymin_ = area[1]
                    xmax_ = area[2]
                    ymax_ = area[3]
                        
                    n_x = 1.5
                    n_y = 1.2

                    xmin = xmin_ - (n_x-1)*(xmax_-xmin_)
                    ymin = ymin_ - (n_y-1)*(ymax_-ymin_)
                    xmax = xmax_ + (n_x-1)*(xmax_-xmin_)
                    ymax = ymax_ + (n_y-1)*(ymax_-ymin_)

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
                tframe_4 = torch.Tensor([])
                tframe_5 = torch.Tensor([])
                tframe_t = torch.Tensor([])
                
                for i, area in enumerate(new_areas):
                    crop_img_1 = (TT(img_1.crop(area).resize((input_size,input_size))).view([1,3,input_size,input_size])*2)-1
                    crop_img_2 = (TT(img_2.crop(area).resize((input_size,input_size))).view([1,3,input_size,input_size])*2)-1
                    crop_img_4 = (TT(img_4.crop(area).resize((input_size,input_size))).view([1,3,input_size,input_size])*2)-1
                    crop_img_5 = (TT(img_5.crop(area).resize((input_size,input_size))).view([1,3,input_size,input_size])*2)-1
                    crop_img_t = (TT(img_t.crop(area).resize((input_size,input_size))).view([1,3,input_size,input_size])*2)-1


                    tframe_1 = torch.cat([tframe_1,crop_img_1],0)
                    tframe_2 = torch.cat([tframe_2,crop_img_2],0)
                    tframe_4 = torch.cat([tframe_4,crop_img_4],0)
                    tframe_5 = torch.cat([tframe_5,crop_img_5],0)
                    tframe_t = torch.cat([tframe_t,crop_img_t],0)
                
                tframe_1 = torch.cat([tframe_1,frame_1 ],0)
                tframe_2 = torch.cat([tframe_2,frame_2 ],0)
                tframe_4 = torch.cat([tframe_4,frame_4 ],0)
                tframe_5 = torch.cat([tframe_5,frame_5 ],0)
                tframe_t = torch.cat([tframe_t,f_target],0)


                # if step % 20 == 0:
                #     img_1.save(f'crop_imgs/tester.png')
                #     save_image(((tframe_1+1)/2),f'crop_imgs/tester_cat_11.png')
                    # print(areas)
                    # save_image(((tframe_2+1)/2),f'crop_imgs/tester_cat_22.png')
                    # save_image(((tframe_3+1)/2),f'crop_imgs/tester_cat_33.png')
                    # save_image(((tframe_4+1)/2),f'crop_imgs/tester_cat_44.png')

                bs_size = len(tframe_1)
                frame_1 = tframe_1.cuda()
                frame_2 = tframe_2.cuda()
                frame_4 = tframe_4.cuda()
                frame_5 = tframe_5.cuda()
                f_target = tframe_t.cuda()

                f_input = torch.cat([frame_1,frame_2, frame_4, frame_5], 1)

                FG_frame = generator(f_input)

                Fgt_flow_input = torch.cat([frame_4.unsqueeze(2), f_target.unsqueeze(2)], 2)
                Fpred_flow_input = torch.cat([frame_4.unsqueeze(2), FG_frame.unsqueeze(2)], 2)

                Bgt_flow_input = torch.cat([frame_2.unsqueeze(2), f_target.unsqueeze(2)], 2)
                Bpred_flow_input = torch.cat([frame_2.unsqueeze(2), FG_frame.unsqueeze(2)], 2)

                F_flow_gt = (flow_net(Fgt_flow_input * 255.) / 255.).detach()
                F_flow_pred = (flow_net(Fpred_flow_input * 255.) / 255.).detach()


                B_flow_gt = (flow_net(Bgt_flow_input * 255.) / 255.).detach()
                B_flow_pred = (flow_net(Bpred_flow_input * 255.) / 255.).detach()
                
                F_fl_l = flow_loss(F_flow_pred, F_flow_gt)
                B_fl_l = flow_loss(B_flow_pred, B_flow_gt)

                inte_l = intensity_loss(FG_frame, f_target)
                grad_l = gradient_loss(FG_frame, f_target)
                
                d_out = discriminator(FG_frame)
            
                g_l = adversarial_loss(d_out)

                G_l = 1. * inte_l + 1. * grad_l + 0.05 * g_l + 2. * (F_fl_l + B_fl_l)
                
                d_ft = discriminator(f_target)
                d_out_d  = discriminator(FG_frame.detach())
                D_l = discriminate_loss(d_ft, d_out_d)

                # Or just do .step() after all the gradients have been computed, like the following way:
                optimizer_G.zero_grad()
                G_l.backward()
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

                
                psnr = psnr_error(FG_frame, f_target).cpu().detach()
                

                if step != start_iter:
                    if step % 20 == 0:
                        time_remain = (train_cfg.iters - step) * iter_t
                        eta = str(datetime.timedelta(seconds=time_remain)).split('.')[0]
                        
                        psnr = psnr_error(FG_frame, f_target).cpu().detach()

                        lr_g = optimizer_G.param_groups[0]['lr']
                        lr_d = optimizer_D.param_groups[0]['lr']


                        print(f"[{step}]  grad_l: {grad_l:.3f} | inte_l: {inte_l:.3f}| g_l: {g_l:.3f}"
                            f"| G_l: {G_l:.3f} | D_fl: {D_l:.3f}"
                            f"| psnr: {psnr:.3f} | F_fl_l: {F_fl_l:.3f}| ETA: {eta} | iter: {iter_t:.3f}s")

                        writer.add_scalar('psnr/forward/train_psnr', psnr, global_step=step)
                        writer.add_scalar('total_loss/forward/g_loss_total', G_l, global_step=step)
                        writer.add_scalar('total_loss/forward/d_loss', D_l, global_step=step)
                        writer.add_scalar('G_loss_total/forward/g_loss', g_l, global_step=step)

                        writer.add_scalar('G_loss_total/forward/inte_loss', inte_l, global_step=step)
                        writer.add_scalar('G_loss_total/forward/grad_loss', grad_l, global_step=step)


                    if step % train_cfg.save_interval == 0:
                        model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                                    'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict()}
                        torch.save(model_dict, f'weights/target_{train_cfg.model}_{train_cfg.dataset}_mid_{input_size}_{step}.pth')
                        print(f'\nAlready saved: \'target_{train_cfg.model}_{train_cfg.dataset}_mid_{input_size}_{step}.pth\'.')

                    if step % train_cfg.val_interval == 0:
                        val_psnr = val(train_cfg, model=generator, dis_model=discriminator, iters=step)
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
