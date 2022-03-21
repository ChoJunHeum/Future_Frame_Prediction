import os
from glob import glob
import cv2
import time
import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import argparse
import random

from utils import *
from losses import *
import Dataset
from models.unet import UNet
from models.vgg16_unet import *
from models.pix2pix_networks import PixelDiscriminator
from models.convLSTM_networks import ConvLstmGenerator
# from models.liteFlownet import lite_flownet as lite_flow
from config import update_config
# from models.flownet2.models import FlowNet2SD
from evaluate_ft import val
from torchvision.utils import save_image

from fid_score import *

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--model', default='ConvLSTM', type=str)
parser.add_argument('--dataset', default='CalTech', type=str, help='The name of the dataset to train.')
parser.add_argument('--iters', default=60000, type=int, help='The total iteration number.')
parser.add_argument('--resume', default=None, type=str,
                    help='The pre-trained model to resume training with, pass \'latest\' or the model name.')
parser.add_argument('--save_interval', default=1000, type=int, help='Save the model every [save_interval] iterations.')
parser.add_argument('--val_interval', default=1000, type=int,
                    help='Evaluate the model every [val_interval] iterations, pass -1 to disable.')


args = parser.parse_args()
train_cfg = update_config(args, mode='train')
train_cfg.print_cfg()


if train_cfg.model == 'ConvLSTM':
    generator = ConvLstmGenerator().cuda()
elif train_cfg.model == 'vgg16bn_unet':
    generator = vgg16bn_unet().cuda()

# print(generator)
# quit()
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

try:
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
            FG_frame = generator(f_input)
            # print(f_input.shape)
            # print(FG_frame.shape)

            inte_fl = intensity_loss(FG_frame, f_target)
            grad_fl = gradient_loss(FG_frame, f_target)

            g_fl = adversarial_loss(discriminator(FG_frame))
            G_fl_t = 1. * inte_fl + 1. * grad_fl + 0.05 * g_fl

            # When training discriminator, don't train generator, so use .detach() to cut off gradients.
            D_fl = discriminate_loss(discriminator(f_target), discriminator(FG_frame.detach()))

            # Backward
            b_input = torch.cat([FG_frame.detach(), frame_4, frame_3, frame_2], 1)
            b_target = frame_1

            BG_frame = generator(b_input)

            inte_bl = intensity_loss(BG_frame, b_target)
            grad_bl = gradient_loss(BG_frame, b_target)

            g_bl = adversarial_loss(discriminator(BG_frame))
            G_bl_t = 1. * inte_bl + 1. * grad_bl + 0.05 * g_bl

            # When training discriminator, don't train generator, so use .detach() to cut off gradients.
            D_bl = discriminate_loss(discriminator(b_target), discriminator(BG_frame.detach()))

            # Total Loss
            inte_l = inte_fl + inte_bl
            grad_l = grad_fl + grad_bl

            g_l = g_fl + g_bl
            G_l_t = G_fl_t + G_bl_t

            D_l = D_fl + D_bl

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
                    
                    f_psnr = psnr_error(FG_frame, f_target)
                    b_psnr = psnr_error(BG_frame, b_target)

                    psnr = (f_psnr + b_psnr)/2


                    lr_g = optimizer_G.param_groups[0]['lr']
                    lr_d = optimizer_D.param_groups[0]['lr']


                    print(f"[{step}]  inte_fl: {inte_fl:.3f} | inte_bl: {inte_bl:.3f} | grad_fl: {grad_fl:.3f} | grad_bl: {grad_bl:.3f} | "
                        f"g_fl: {g_fl:.3f} | g_bl: {g_bl:.3f} | G_fl_total: {G_fl_t:.3f} | G_bl_total: {G_bl_t:.3f} | D_fl: {D_fl:.3f} | D_bl: {D_bl:.3f} | "
                        f"| f_psnr: {f_psnr:.3f} | b_psnr: {b_psnr:.3f} | ETA: {eta} | iter: {iter_t:.3f}s")

                    save_FG_frame = ((FG_frame[0] + 1) / 2)
                    save_FG_frame = save_FG_frame.cpu().detach()[(2, 1, 0), ...]
                    save_F_target = ((f_target[0] + 1) / 2)
                    save_F_target = save_F_target.cpu().detach()[(2, 1, 0), ...]

                    save_BG_frame = ((BG_frame[0] + 1) / 2)
                    save_BG_frame = save_BG_frame.cpu().detach()[(2, 1, 0), ...]
                    save_B_target = ((b_target[0] + 1) / 2)
                    save_B_target = save_B_target.cpu().detach()[(2, 1, 0), ...]

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

                if step % 1000 == 0:
                    save_image(save_FG_frame, f'training_imgs/{data_name}/{step}_FG_frame.png')
                    save_image(save_F_target, f'training_imgs/{data_name}/{step}_FT_frame_.png')
                    
                    save_image(save_BG_frame, f'training_imgs/{data_name}/{step}_BG_frame_.png')
                    save_image(save_B_target, f'training_imgs/{data_name}/{step}_BT_frame_.png')

                if step % int(train_cfg.iters / 100) == 0:
                    writer.add_image('image/FG_frame', save_FG_frame, global_step=step)
                    writer.add_image('image/f_target', save_F_target, global_step=step)

                    writer.add_image('image/BG_frame', save_BG_frame, global_step=step)
                    writer.add_image('image/b_target', save_B_target, global_step=step)


                if step % train_cfg.save_interval == 0:
                    model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                                'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict()}
                    torch.save(model_dict, f'weights/lstm_{train_cfg.dataset}_{step}.pth')
                    print(f'\nAlready saved: \'lstm_{train_cfg.dataset}_{step}.pth\'.')
                    

            step += 1
            if step > train_cfg.iters:
                training = False
                model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                            'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict()}
                torch.save(model_dict, f'weights/latest_{train_cfg.dataset}_{step}.pth')
                break

except KeyboardInterrupt:
    print(f'\nStop early, model saved: \'latest_{train_cfg.dataset}_{step}.pth\'.\n')

    if glob(f'weights/latest*'):
        os.remove(glob(f'weights/latest*')[0])

    model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                  'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict()}
    torch.save(model_dict, f'weights/latest_{train_cfg.dataset}_{step}.pth')
