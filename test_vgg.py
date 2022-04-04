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

from util import *
from losses import *
import Dataset
from rl_utils import *
from model.RL_model import *
from model.unet import UNet
from model.vgg16_unet import *

from model.pix2pix_networks import PixelDiscriminator
from ft_config import update_config
from evaluate_ft import val
from torchvision.utils import save_image
from torchsummary import summary as summary_

generator = vgg16bn_unet()

discriminator = PixelDiscriminator(input_nc=3).cuda()
print(discriminator)