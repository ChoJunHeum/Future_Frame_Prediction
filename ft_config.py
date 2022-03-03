#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from glob import glob
import os

if not os.path.exists('tensorboard_log'):
    os.mkdir('tensorboard_log')
if not os.path.exists('weights'):
    os.mkdir('weights')
if not os.path.exists('results'):
    os.mkdir('results')

share_config = {'mode': 'training',
                'dataset': 'avenue',
                'img_size': (256, 256),
                'data_root': '/home/chojh21c/ADGW/Future_Frame_Prediction/datasets/'}  # remember the final '/'


class dict2class:
    def __init__(self, config):
        for k, v in config.items():
            self.__setattr__(k, v)

    def print_cfg(self):
        print('\n' + '-' * 30 + f'{self.mode} cfg' + '-' * 30)
        for k, v in vars(self).items():
            print(f'{k}: {v}')
        print()


def update_config(args=None, mode=None):
    share_config['mode'] = mode
    assert args.dataset in ('ped2', 'avenue', 'shanghai', 'CalTech'), 'Dataset error.'
    share_config['dataset'] = args.dataset

    if mode == 'train':
        share_config['batch_size'] = args.batch_size
        share_config['train_data'] = share_config['data_root'] + args.dataset + '/training/'
        share_config['d_lr'] = 0.00001
        share_config['r_lr'] = 0.00002
        share_config['ft_g_lr'] = 0.00001
        share_config['resume_g'] = glob(f'weights/{args.resume_g}*')[0] if args.resume_g else None
        share_config['resume_r'] = glob(f'weights/{args.resume_r}*')[0] if args.resume_r else None
        share_config['save_interval'] = args.save_interval

        share_config['iters'] = args.iters


    elif mode == 'test':
        share_config['test_data'] = share_config['data_root'] + args.dataset + '/testing/'
        share_config['model'] = args.model

    return dict2class(share_config)  # change dict keys to class attributes
