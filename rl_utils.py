import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from utils import *
from torch.autograd import Variable
from losses import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition', ('state','target', 'action', 'next_state','next_target','reward', 'episode','step', 'cor'))

m = MSSSIM()
gradient_loss = Gradient_Loss(3).cuda()
intensity_loss = Intensity_Loss().cuda()

class Env():
    def __init__ (self):
        self.state = None

    # image quality 스코어로 리턴하면 됨. 근데 뭐로하지?
    def step_R(self, action, input, target, answer):
        
        total_len = len(action)
        cor = (action == answer)

        true_cor = 0
        false_cor = 0

        psnr = psnr_error_ft(input, target)
        # psnr = psnr_error(input, target)
        
        ssim_score = msssim(input, target)

        if answer:
            reward = torch.mul(cor, 2)
            true_cor = true_cor + sum(action == answer).item()

        else:
            reward = torch.mul(cor, .8)
            false_cor = false_cor + sum(action == answer).item()

        cor_sum = sum(cor).item()

        # print(action)
        # print(true_cor, total_cor, false_cor)
        reward = reward - .5

        return reward, cor_sum, total_len, psnr, true_cor, false_cor, cor

    # image quality 스코어로 리턴하면 됨. 근데 뭐로하지?
    def step_G(self, action, answer):

        cor = (action == answer)
        total_cor = sum(cor).item()
        return cor 

    # data -> state
    def reset(self, data):
        self.state = data
        return data


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
