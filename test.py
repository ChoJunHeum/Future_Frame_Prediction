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
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--dataset', default='CalTech', type=str, help='The name of the dataset to train.')
parser.add_argument('--iters', default=60000, type=int, help='The total iteration number.')
parser.add_argument('--resume_g', default=None, type=str, help='The pre-trained generator model to finetuning with.')
parser.add_argument('--resume_r', default=None, type=str, help='The pre-trained RL model to training with.')
parser.add_argument('--save_interval', default=1000, type=int, help='Save the model every [save_interval] iterations.')

###
torch.autograd.set_detect_anomaly(True)
###


args = parser.parse_args()
train_cfg = update_config(args, mode='train')
train_cfg.print_cfg()

generator = UNet(input_channels=12, output_channel=3).cuda()

policy_net = Agent().cuda()
target_net = Agent().cuda()
target_net.load_state_dict(policy_net.state_dict())

optimizer_G = torch.optim.Adam(generator.parameters(), lr=train_cfg.ft_g_lr)
optimizer_R = torch.optim.Adam(policy_net.parameters(), lr=train_cfg.r_lr)

memory = ReplayMemory(32)

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
    optimizer_R.load_state_dict(torch.load(train_cfg.resume_r)['optimizer_r'])
    print(f'Pre-trained RL model has been loaded.\n')
else:
    policy_net.apply(weights_init_normal)
    target_net.load_state_dict(policy_net.state_dict())
    print('RL model is going to be trained from scratch.\n')


Transition = namedtuple('Transition', ('state','target', 'action', 'next_state','next_target','reward', 'episode','step'))
GAMMA = 0.98
TARGET_UPDATE = 20
batch_size = train_cfg.batch_size
action_rand = 0.2

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

epi_reward = 0
epi_cor = 0
epi_count = 0
epi_iq = 0
epi_true = 0
epi_false = 0
epi_true_cor = 0
epi_false_cor = 0

with torch.autograd.detect_anomaly():

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

                targets = []

                for i in range(4,clip_len):
                    targets.append(clips[:, i*3:(i+1)*3, :, :].cuda())
                
                for index in indice:
                    if len(train_dataset.all_seqs[index]) == 0:
                        train_dataset.all_seqs[index] = list(range(len(train_dataset.videos[index]) - 4))
                        random.shuffle(train_dataset.all_seqs[index])

                G_frame = generator(input_frames)
                G_frame_detached = G_frame.detach()
                # cur_state = torch.cat([input_frames, G_frame_detached], 1)
                cur_state = torch.cat([input_frames, G_frame], 1)
                action_input = torch.cat([input_frames, G_frame_detached], 1)
                
                # episode 시작
                for i, target in enumerate(targets):
                    answer = len(targets)==(i+1)

                    random_num = random.random()
                    if(random_num > action_rand):
                        action = policy_net(action_input, target).max(1)[1]
                    else:
                        action = torch.randint(2,(2,)).cuda()

                    reward, cor, count, psnr, true_cor, false_cor = env.step_R(action, G_frame, target, answer)
            
                    epi_reward = epi_reward + reward.sum().item()
                    epi_cor = epi_cor + cor
                    epi_true_cor = epi_true_cor + true_cor
                    epi_false_cor = epi_false_cor + false_cor

                    if answer:
                        epi_true = epi_true + count
                    else:
                        epi_false = epi_false + count

                    epi_count = epi_count + count
                    epi_iq = epi_iq + psnr.sum().item()

                    # print("----------------------------------")
                    # print("Answer: ", answer)
                    # print("reward: ", reward)
                    # print("action: ", action)

                    # 다음 state
                    input_frames = torch.cat((input_frames[:,3:12,:,:], G_frame_detached), 1)
                    G_frame = generator(input_frames)
                    G_frame_detached = G_frame.detach()

                    if len(targets)!=(i+1):
                        next_target = targets[i+1]
                        next_state = torch.cat([input_frames, G_frame_detached], 1)
                    else:
                        next_target = None
                        next_state = None
                    
                    memory.push(cur_state, target, action, next_state, next_target, reward, step, i)
                    cur_state = torch.cat([input_frames, G_frame], 1)
                    action_input = torch.cat([input_frames, G_frame_detached], 1)
                    # print(f'mem pushed, current len: {len(memory)}')

                # optimize_model
                if step % 10 == 0:

                    rwd = epi_reward/epi_count
                    acc = (epi_cor/epi_count)*100
                    true_acc = (epi_true_cor/epi_true)*100
                    false_acc = (epi_false_cor/epi_false)*100

                    if(epi_cor!=0):
                        iq = epi_iq/epi_count
                    else:
                        iq = 0

                    print(f"{step} | Start Optimizing | Reward: {rwd:.2f} | Accuracy: {acc:.2f} | Image Quality: {iq:.2f}")
                    
                    writer.add_scalar('finetuning/reward', rwd, global_step=step)
                    writer.add_scalar('finetuning/accracy', acc, global_step=step)
                    writer.add_scalar('finetuning/image_quality', iq, global_step=step)

                    # len(memory) : # of episodes
                    transitions = memory.sample(2)

                    batch = Transition(*zip(*transitions))
                    
                    # batch.state: tuple[memory_sampling_size, batch_size, 3*5, 256, 256])           ex: 8,4,15,256,256
                    # batch.target: tuple[memory_sampling_size, batch_size, 3, 256, 256])            ex: 8,4,3,256,256
                    # batch.action: tuple[memory_sampling_size, batch_size])                         ex: 8,4
                    # batch.reward: tuple[memory_sampling_size, batch_size])                         ex: 8,4
                    # batch.next_state: [memory_sampling_size, batch_size, 3*5, 256, 256])           ex: 8,4,15,256,256
                    # batch.next_target: [memory_sampling_size, batch_size, 3, 256, 256])            ex: 8,4,3,256,256

                    # Compute a mask of non-final states and concatenate the batch elements
                    # (a final state would've been the one after which simulation ended)
                    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                        batch.next_state)), device=device, dtype=torch.bool)
                    
                    if non_final_mask.sum().item() != 0:
                        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
                        non_final_next_target = torch.cat([s for s in batch.next_target if s is not None])

                        # mask 차원 변경
                        temp_mask = non_final_mask.view(-1,1)
                        temp_mask = torch.cat([temp_mask]*batch_size,1)
                        non_final_mask = temp_mask.view(-1)

                        state_batch = torch.cat(batch.state)
                        target_batch = torch.cat(batch.target)
                        action_batch = torch.cat(batch.action)
                        reward_batch = torch.cat(batch.reward)
                        next_state_batch = torch.cat([s for s in batch.next_state if s is not None])


                        episode_batch = batch.episode
                        step_batch = batch.step

                        # state_batch: torch.Size([batch_size * memory_sampling_size, 3*5, 256, 256])   ex: 32,15,256,256
                        # target_batch: torch.Size([batch_size * memory_sampling_size, 3, 256, 256])    ex: 32,3,256,256
                        # action_batch: torch.Size([batch_size * memory_sampling_size])                 ex: 32
                        # reward_batch: torch.Size([batch_size * memory_sampling_size])                 ex: 32
                        # non_final_next_states: torch.Size([batch_size * memory_sampling_size - final_episode])                 ex: 28,15,256,256
                        # non_final_next_target: torch.Size([batch_size * memory_sampling_size - final_episode])                 ex: 28,3,256,256

                        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                        # columns of actions taken. These are the actions which would've been taken
                        # for each batch state according to policy_net
                        # state_action_values = policy_net(state_batch, target_batch).gather(1, action_batch)
                        # state_action_values's shape: torch.Size([batch_size * memory_sampling_size, Q(s,a)]) ex: 32,1
                        state_action_values = policy_net(state_batch, target_batch).gather(1, action_batch.view(-1,1))


                        # Compute V(s_{t+1}) for all next states.
                        # Expected values of actions for non_final_next_states are computed based
                        # on the "older" target_net; selecting their best reward with max(1)[0].
                        # This is merged based on the mask, such that we'll have either the expected
                        # state value or 0 in case the state was final.
                        next_state_values = torch.zeros(len(non_final_mask), device=device)
                        
                        # next_state_values[non_final_mask] = target_net(non_final_next_states, non_final_next_target).max(1)[:,0].detach()
                        # print("nfm: ",non_final_mask)
                        # print("nsv: ",target_net(non_final_next_states, non_final_next_target))

                        next_state_values[non_final_mask] = target_net(non_final_next_states, non_final_next_target).max(1)[0].detach()

                        # Compute the expected Q values
                        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

                        # print("esav\n",expected_state_action_values.unsqueeze(1).shape)

                        # print("sav\n", state_action_values)

                        # Compute Huber loss
                        criterion = nn.SmoothL1Loss()
                        loss_R = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
                        # loss_G = loss_R.clone()
                        # print("Loss: ", loss_R)
                        # print("episode: ", episode)
                        # print("step: ", step)

                        for e, s in zip(episode_batch, step_batch):
                            print(f"{e}-{s} | ")
                        # Optimize the model
                        
                        optimizer_R.zero_grad()
                        optimizer_G.zero_grad()

                        loss_R.backward(retain_graph=True)
                        optimizer_R.step()
                    
                        for param in policy_net.parameters():
                            param.grad.data.clamp_(-1, 1)
                        # loss_G.backward()
                        optimizer_G.step()


                    epi_reward = 0
                    epi_cor = 0
                    epi_count = 0
                    epi_iq = 0
                    epi_true = 0
                    epi_false = 0

                    epi_true_cor = 0
                    epi_false_cor = 0
                
                action_rand = action_rand*0.999
                        
                if step % train_cfg.save_interval == 0:
                    model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                            'net_r': target_net.state_dict(), 'optimizer_r': optimizer_R.state_dict()}
                    print(f'\nAlready saved: \'ft_{train_cfg.resume_g}_{step}.pth\'.')

                if step % TARGET_UPDATE == 0:
                    print("Update Target Network.")
                    target_net.load_state_dict(policy_net.state_dict())

                step += 1
                if step > train_cfg.iters:
                    training = False
                    model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                                'net_r': target_net.state_dict(), 'optimizer_r': optimizer_R.state_dict()}
                    torch.save(model_dict, f'weights/ft_{train_cfg.resume_g}_{step}.pth')
                    break


    except KeyboardInterrupt:
        print(f'\nStop early, model saved: \'latest_{train_cfg.dataset}_{step}.pth\'.\n')

        # if glob(f'weights/latest*'):
        #     os.remove(glob(f'weights/latest*')[0])

        # model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
        #               'net_r': target_net.state_dict(), 'optimizer_r': optimizer_R.state_dict()}
        # torch.save(model_dict, f'weights/ft_{train_cfg.dataset}_{step}.pth')
