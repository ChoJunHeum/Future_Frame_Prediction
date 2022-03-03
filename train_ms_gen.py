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
# from models.liteFlownet import lite_flownet as lite_flow
from config import update_config
# from models.flownet2.models import FlowNet2SD
from evaluate_ft import val
from torchvision.utils import save_image

from fid_score import *

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--batch_size', default=4, type=int)
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

adversarial_loss = Adversarial_Loss().cuda()
discriminate_loss = Discriminate_Loss().cuda()
gradient_loss = Gradient_Loss(3).cuda()
intensity_loss = Intensity_Loss().cuda()

train_dataset = Dataset.ms_dataset(train_cfg)

# Remember to set drop_last=True, because we need to use 4 frames to predict one frame.
train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_cfg.batch_size,
                              shuffle=True, num_workers=4, drop_last=True)


writer = SummaryWriter(f'tensorboard_log/ms_{train_cfg.dataset}_bs{train_cfg.batch_size}')
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

            epi_iq = 0

            # target_pos: 6 ~ 9
            # 0,1,2,3 : input
            # 4,5,6,7,8 : target

            clip_len = 9
            tar_len = clip_len - 4

            targets = []

            for i in range(4,9):
                targets.append(clips[:, i*3:(i+1)*3, :, :].cuda())

            for index in indice:
                if len(train_dataset.all_seqs[index]) == 0:
                    train_dataset.all_seqs[index] = list(range(len(train_dataset.videos[index]) - 4))
                    random.shuffle(train_dataset.all_seqs[index])

            input_frames = clips[:, 0:12, :, :].cuda()
            FG_frame = generator(input_frames)
            
            f_psnrs = 0
            b_psnrs = 0
            psnr_count = 0

            # episode 시작
            for i, target in enumerate(targets):

                frame_1 = input_frames[:, 0:3, :, :].cuda()  # (n, 12, 256, 256) 
                frame_2 = input_frames[:, 3:6, :, :].cuda()  # (n, 12, 256, 256) 
                frame_3 = input_frames[:, 6:9, :, :].cuda()  # (n, 12, 256, 256) 
                frame_4 = input_frames[:, 9:12, :, :].cuda()  # (n, 12, 256, 256) 
                f_target = target

                f_input = torch.cat([frame_1,frame_2, frame_3, frame_4], 1)

                # pop() the used frame index, this can't work in train_dataset.__getitem__ because of multiprocessing.
                for index in indice:
                    train_dataset.all_seqs[index].pop()
                    if len(train_dataset.all_seqs[index]) == 0:
                        train_dataset.all_seqs[index] = list(range(len(train_dataset.videos[index]) - 4))
                        random.shuffle(train_dataset.all_seqs[index])

                # Forward
                FG_frame = generator(f_input)

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

                if step % 20 == 0:
                    f_psnrs = f_psnrs + psnr_error(FG_frame, f_target)
                    b_psnrs = b_psnrs + psnr_error(BG_frame, b_target)
                    psnr_count = psnr_count + 1


                if step % 1000 == 0:

                    save_FG_frame = ((FG_frame[0] + 1) / 2)
                    save_FG_frame = save_FG_frame.cpu().detach()[(2, 1, 0), ...]
                    save_F_target = ((f_target[0] + 1) / 2)
                    save_F_target = save_F_target.cpu().detach()[(2, 1, 0), ...]

                    save_BG_frame = ((BG_frame[0] + 1) / 2)
                    save_BG_frame = save_BG_frame.cpu().detach()[(2, 1, 0), ...]
                    save_B_target = ((b_target[0] + 1) / 2)
                    save_B_target = save_B_target.cpu().detach()[(2, 1, 0), ...]

                    save_image(save_FG_frame, f'training_imgs/{data_name}/{step}_{i}_FG_frame.png')
                    save_image(save_F_target, f'training_imgs/{data_name}/{step}_{i}_FT_frame_.png')

                    save_image(save_BG_frame, f'training_imgs/{data_name}/{step}_{i}_FG_frame.png')
                    save_image(save_B_target, f'training_imgs/{data_name}/{step}_{i}_FT_frame_.png')


                # 다음 frame
                input_frames = torch.cat((input_frames[:,3:12,:,:], FG_frame.detach()), 1)
                FG_frame = generator(input_frames)


            if step != start_iter:
                if step % 20 == 0:
                    time_remain = (train_cfg.iters - step) * iter_t
                    eta = str(datetime.timedelta(seconds=time_remain)).split('.')[0]

                    f_psnr = f_psnrs/psnr_count
                    b_psnr = b_psnrs/psnr_count
                    


                    print(f"[{step}]  inte_fl: {inte_fl:.3f} | inte_bl: {inte_bl:.3f} | grad_fl: {grad_fl:.3f} | grad_bl: {grad_bl:.3f} | "
                        f"g_fl: {g_fl:.3f} | g_bl: {g_bl:.3f} | G_fl_total: {G_fl_t:.3f} | G_bl_total: {G_bl_t:.3f} | D_fl: {D_fl:.3f} | D_bl: {D_bl:.3f} | "
                        f"| f_psnr: {f_psnr:.3f} | b_psnr: {b_psnr:.3f} | ETA: {eta} | iter: {iter_t:.3f}s")

                    writer.add_scalar('psnr/forward/train_psnr', f_psnr, global_step=step)
                    writer.add_scalar('total_loss/forward/g_loss_total', G_fl_t, global_step=step)
                    writer.add_scalar('total_loss/forward/d_loss', D_fl, global_step=step)
                    writer.add_scalar('G_loss_total/forward/g_loss', g_fl, global_step=step)

                    writer.add_scalar('G_loss_total/forward/inte_loss', inte_fl, global_step=step)
                    writer.add_scalar('G_loss_total/forward/grad_loss', grad_fl, global_step=step)

                if step % 1000 == 0:

                    save_FG_frame = ((FG_frame[0] + 1) / 2)
                    save_FG_frame = save_FG_frame.cpu().detach()[(2, 1, 0), ...]
                    save_F_target = ((f_target[0] + 1) / 2)
                    save_F_target = save_F_target.cpu().detach()[(2, 1, 0), ...]

                    save_BG_frame = ((BG_frame[0] + 1) / 2)
                    save_BG_frame = save_BG_frame.cpu().detach()[(2, 1, 0), ...]
                    save_B_target = ((b_target[0] + 1) / 2)
                    save_B_target = save_B_target.cpu().detach()[(2, 1, 0), ...]

                    save_image(save_FG_frame, f'training_imgs/{data_name}/{step}_FG_frame.png')
                    save_image(save_F_target, f'training_imgs/{data_name}/{step}_FT_frame_.png')

                    save_image(save_BG_frame, f'training_imgs/{data_name}/{step}_FG_frame.png')
                    save_image(save_B_target, f'training_imgs/{data_name}/{step}_FT_frame_.png')

                if step % train_cfg.save_interval == 0:
                    model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                                'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict()}
                    torch.save(model_dict, f'weights/{train_cfg.dataset}_{step}.pth')
                    print(f'\nAlready saved: \'vgg_{train_cfg.dataset}_{step}.pth\'.')


                if step % train_cfg.save_interval == 0:
                    model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                                'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict()}
                    torch.save(model_dict, f'weights/ms_{train_cfg.dataset}_{step}.pth')
                    print(f'\nAlready saved: \'vgg_{train_cfg.dataset}_{step}.pth\'.')
                    

            step += 1
            if step > train_cfg.iters:
                training = False
                model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                            'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict()}
                torch.save(model_dict, f'weights/ms_latest_{train_cfg.dataset}_{step}.pth')
                break

except KeyboardInterrupt:
    print(f'\nStop early, model saved: \'latest_{train_cfg.dataset}_{step}.pth\'.\n')

    if glob(f'weights/latest*'):
        os.remove(glob(f'weights/latest*')[0])

    model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                  'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict()}
    torch.save(model_dict, f'weights/ms_latest_{train_cfg.dataset}_{step}.pth')
