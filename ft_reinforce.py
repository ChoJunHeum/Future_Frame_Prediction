import os
from glob import glob
import cv2
import time
import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import argparse
import random
import time

from utils import *
from losses import *
import Dataset
from rl_utils import *
from models.RL_model import *
from models.unet import UNet
from models.pix2pix_networks import PixelDiscriminator
from ft_config import update_config
from evaluate_ft import val
from torchvision.utils import save_image

from fid_score import *

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--dataset', default='CalTech', type=str, help='The name of the dataset to train.')
parser.add_argument('--iters', default=60000, type=int, help='The total iteration number.')
parser.add_argument('--resume_g', default=None, type=str, help='The pre-trained generator model to finetuning with.')
parser.add_argument('--resume_r', default=None, type=str, help='The pre-trained RL model to training with.')
parser.add_argument('--save_interval', default=5000, type=int, help='Save the model every [save_interval] iterations.')


args = parser.parse_args()
train_cfg = update_config(args, mode='train')
train_cfg.print_cfg()

generator = UNet(input_channels=12, output_channel=3).cuda()

policy_net = Agent().cuda()
target_net = Agent().cuda()
target_net.load_state_dict(policy_net.state_dict())

optimizer_G = torch.optim.Adam(generator.parameters(), lr=train_cfg.ft_g_lr)

env = Env()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if train_cfg.resume_g:
    generator.load_state_dict(torch.load(train_cfg.resume_g)['net_g'])
    optimizer_G.load_state_dict(torch.load(train_cfg.resume_g)['optimizer_g'])
    print(f'Pre-trained generator has been loaded.\n')
else:
    generator.apply(weights_init_normal)
    print('Generator is going to be trained from scratch.\n')

if train_cfg.resume_r:
    policy_net.load_state_dict(torch.load(train_cfg.resume_r)['net_r'])
    target_net.load_state_dict(policy_net.state_dict())
    print(f'Pre-trained RL model has been loaded.\n')
else:
    policy_net.apply(weights_init_normal)
    target_net.load_state_dict(policy_net.state_dict())
    print('RL model is going to be trained from scratch.\n')


GAMMA = 0.98
batch_size = train_cfg.batch_size

train_dataset = Dataset.ft_dataset(train_cfg)

# Remember to set drop_last=True, because we need to use 4 frames to predict one frame.
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, drop_last=True)

writer = SummaryWriter(f'tensorboard_log/ft_{train_cfg.dataset}_bs{batch_size}')
training = True
generator = generator.train()

data_name = args.dataset

# total_reward = 0
# total_cor = 0
# total_count = 0
with torch.autograd.set_detect_anomaly(True):
    try:
        step = 1
        while training:
            for indice, clips, target_pos in train_dataloader:

                # target_pos: 6 ~ 9
                # 0,1,2,3 : input
                # 4,5,6,7,8 : target

                clip_len = target_pos[0]
                tar_len = clip_len - 4
                input_frames = clips[:, 0:12, :, :].cuda()

                epi_reward = 0
                epi_cor = 0
                epi_count = 0
                epi_iq = 0
                epi_true = 0
                epi_false = 0

                epi_true_cor = 0
                epi_false_cor = 0

                targets = []

                for i in range(4,clip_len):
                    targets.append(clips[:, i*3:(i+1)*3, :, :].cuda())
                
                for index in indice:
                    if len(train_dataset.all_seqs[index]) == 0:
                        train_dataset.all_seqs[index] = list(range(len(train_dataset.videos[index]) - 4))
                        random.shuffle(train_dataset.all_seqs[index])

                G_frame = generator(input_frames)
                cur_state = torch.cat([input_frames, G_frame], 1)
                
                # episode 시작
                for i, target in enumerate(targets):

                    answer = len(targets)==(i+1)
                    random_num = random.random()

                    # Select action 
                    if(random_num > 0.1):
                        action = policy_net(cur_state, target).max(1)[1]
                    else:
                        action = torch.randint(2,(batch_size,)).cuda()

                    reward, cor, count, psnr, true_cor, false_cor = env.step_R(action, G_frame, target, answer)
                    
                    cor_frame = G_frame[cor]
                    cor_target = target[cor]

                    inte_l = intensity_loss(cor_frame, cor_target)
                    grad_l = gradient_loss(cor_frame, cor_target)

                    loss_G = inte_l + grad_l

                    # Next state input
                    input_frames = torch.cat((input_frames[:,3:12,:,:], G_frame), 1).detach()
                    G_frame = generator(input_frames)
                    cur_state = torch.cat([input_frames, G_frame], 1).detach()
                        
                    # Optimize the model
                    optimizer_G.zero_grad()
                    loss_G.backward()
                    optimizer_G.step()
                
                    for param in policy_net.parameters():
                        param.grad.data.clamp_(-1, 1)

                
                if step % 10 == 0:
                    print(f"{step} | Start Optimizing | Image Quality: {iq:.2f} | Loss: {loss_G:.2f}")

                        
                if step % train_cfg.save_interval == 0:
                    model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict()}
                    print(f'\nAlready saved: \'ft_{train_cfg.resume_g}_{step}.pth\'.')

                step += 1
                if step > train_cfg.iters:
                    training = False
                    model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict()}
                    torch.save(model_dict, f'weights/ft_{train_cfg.dataset}_{step}.pth')
                    break

    except KeyboardInterrupt:
        print(f'\nStop early, model saved: \'latest_{train_cfg.dataset}_{step}.pth\'.\n')
        # if glob(f'weights/latest*'):
        #     os.remove(glob(f'weights/latest*')[0])
        # model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
        #               'net_r': target_net.state_dict(), 'optimizer_r': optimizer_R.state_dict()}
        # torch.save(model_dict, f'weights/ft_{train_cfg.dataset}_{step}.pth')
