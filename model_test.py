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


generator = vgg16bn_unet().cuda()

print(generator)