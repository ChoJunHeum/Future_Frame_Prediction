from imghdr import tests
import numpy as np
import os
import time
import torch
import argparse
import cv2
from PIL import Image
import io
from sklearn import metrics
import matplotlib.pyplot as plt
from model.vgg16_unet import *
from model.pix2pix_networks import PixelDiscriminator

from config import update_config
from Dataset import Label_loader
from util import psnr_error, psnr_error_ft
import Dataset
from model.unet import UNet

from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import ToTensor

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--dataset', default='avenue', type=str, help='The name of the dataset to train.')
parser.add_argument('--trained_model', default=None, type=str, help='The pre-trained model to evaluate.')
parser.add_argument('--show_curve', action='store_true',
                    help='Show and save the psnr curve real-timely, this drops fps.')


def val(cfg, model=None):
    if model:  # This is for testing during training.
        generator = model
        generator.eval()
    else:
        generator = vgg16bn_unet().cuda().eval()
        generator.load_state_dict(torch.load('weights/' + cfg.trained_model)['net_g'])
        
        discriminator = PixelDiscriminator(input_nc=3).cuda()
        discriminator.load_state_dict(torch.load('weights/' + cfg.trained_model)['net_d'])

        print(f'The pre-trained generator has been loaded from \'weights/{cfg.trained_model}\'.\n')
        
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True).cuda()


    video_folders = os.listdir(cfg.test_data)
    video_folders.sort()
    video_folders = [os.path.join(cfg.test_data, aa) for aa in video_folders]
    TT = ToTensor()

    fps = 0
    psnr_group_mean = []
    psnr_group_max = []
    psnr_group_min = []

    save_group = []
    score_group = []

    with torch.no_grad():
        for i, folder in enumerate(video_folders):
            dataset = Dataset.test_dataset(cfg, folder)

            psnrs_mean = []
            psnrs_max = []
            psnrs_min = []
            
            dis_scores = []
            save_data = []
            for j, clip in enumerate(dataset):
                input_np = clip[0:12, :, :]
                target_np = clip[12:15, :, :]
                input_frames = torch.from_numpy(input_np).unsqueeze(0).cuda()
                f_target = torch.from_numpy(target_np).unsqueeze(0).cuda()

                frame_1 = input_frames[:,0:3, :, :]  # (n, 12, 256, 256) 
                frame_2 = input_frames[:,3:6, :, :]  # (n, 12, 256, 256) 
                frame_3 = input_frames[:,6:9, :, :]  # (n, 12, 256, 256) 
                frame_4 = input_frames[:,9:12, :, :] # (n, 12, 256, 256) 

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
                img_1.save(f'crop_imgs/tester_eval.png')

                results = yolo_model(img_1)
                areas = results.xyxy[0]
                
                new_areas = []
                for _, area in enumerate(areas):
                    
                    area = area.tolist()

                    if area[4] >0.5:

                        xmin = area[0]
                        ymin = area[1]
                        xmax = area[2]
                        ymax = area[3]
                            
                        n_x = 2
                        n_y = 1.5

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
                
                if len(new_areas) != 0:

                    tframe_1 = torch.Tensor([])
                    tframe_2 = torch.Tensor([])
                    tframe_3 = torch.Tensor([])
                    tframe_4 = torch.Tensor([])
                    tframe_t = torch.Tensor([])
                    
                    for k, area in enumerate(new_areas):
                        crop_img_1 = TT(img_1.crop(area).resize((256,256))).view([1,3,256,256])
                        crop_img_2 = TT(img_2.crop(area).resize((256,256))).view([1,3,256,256])
                        crop_img_3 = TT(img_3.crop(area).resize((256,256))).view([1,3,256,256])
                        crop_img_4 = TT(img_4.crop(area).resize((256,256))).view([1,3,256,256])
                        crop_img_t = TT(img_t.crop(area).resize((256,256))).view([1,3,256,256])

                        tframe_1 = torch.cat([tframe_1,crop_img_1],0)
                        tframe_2 = torch.cat([tframe_2,crop_img_2],0)
                        tframe_3 = torch.cat([tframe_3,crop_img_3],0)
                        tframe_4 = torch.cat([tframe_4,crop_img_4],0)
                        tframe_t = torch.cat([tframe_t,crop_img_t],0)

                        if k % 100 == 0:
                            # crop_img_1_save = ((crop_img_1[0]+1)/2)[(2,1,0),...]
                            # crop_img_2_save = ((crop_img_2[0]+1)/2)[(2,1,0),...]
                            # crop_img_3_save = ((crop_img_3[0]+1)/2)[(2,1,0),...]
                            # crop_img_4_save = ((crop_img_4[0]+1)/2)[(2,1,0),...]
                            # crop_img_t_save = ((crop_img_t[0]+1)/2)[(2,1,0),...]

                            save_image(crop_img_1,f'crop_imgs/tester_1_{k}.png')
                            save_image(crop_img_2,f'crop_imgs/tester_2_{k}.png')
                            save_image(crop_img_3,f'crop_imgs/tester_3_{k}.png')
                            save_image(crop_img_4,f'crop_imgs/tester_4_{k}.png')
                            save_image(crop_img_t,f'crop_imgs/tester_t_{k}.png')
                    
                    frame_1 = tframe_1.cuda()
                    frame_2 = tframe_2.cuda()
                    frame_3 = tframe_3.cuda()
                    frame_4 = tframe_4.cuda()
                    f_target = tframe_t.cuda()

                    f_input = torch.cat([frame_1,frame_2, frame_3, frame_4], 1)
                    G_frame = generator(f_input)
                    save_image(tframe_1, f'img_test/gndk_img.jpg')

                    for n, f in enumerate(G_frame):
                        save_image(f, f'img_test/{n}_img.jpg')

                    # quit()

                    d_out, d_score = discriminator(G_frame)
                    psnr = psnr_error_ft(G_frame, f_target).cpu().detach()
                    # print(type(psnr))
                else:
                    d_score = torch.Tensor([0])
                
                psnrs_mean.append(torch.mean(psnr))
                psnrs_max.append(torch.max(psnr))
                psnrs_min.append(torch.min(psnr))
                
                save_data.append(d_score.cpu().detach().numpy())
                dis_scores.append((torch.min(d_score)).cpu().detach().numpy())

                torch.cuda.synchronize()
                end = time.time()
                if j > 1:  # Compute fps by calculating the time used in one completed iteration, this is more accurate.
                    fps = 1 / (end - temp)
                temp = end
                print(f'\rDetecting: [{i + 1:02d}] {j + 1}/{len(dataset)}, {fps:.2f} fps.', end='')

            # break

            psnr_group_mean.append(np.array(psnrs_mean))
            psnr_group_min.append(np.array(psnrs_min))
            psnr_group_max.append(np.array(psnrs_max))
            

            score_group.append(np.array(dis_scores))

            save_group.append(save_data)

            if not model:
                if cfg.show_curve:
                    video_writer.release()
                    curve_writer.release()


    print('\nAll frames were detected, begin to compute AUC.')

    gt_loader = Label_loader(cfg, video_folders)  # Get gt labels.
    gt = gt_loader()
    print(len(gt))


    assert len(psnr_group_mean) == len(gt), f'Ground truth has {len(gt)} videos, but got {len(psnr_group)} detected videos.'

    scores_mean = np.array([], dtype=np.float32)
    scores_max = np.array([], dtype=np.float32)
    scores_min = np.array([], dtype=np.float32)
    
    labels = np.array([], dtype=np.int8)
    saves = np.array([], dtype=np.float32)
    d_scores = np.array([], dtype=np.float32)
    psnr_source_mean = np.array([], dtype=np.float32)
    psnr_source_min = np.array([], dtype=np.float32)
    psnr_source_max = np.array([], dtype=np.float32)
    

    for i in range(len(psnr_group_mean)):
        distance_mean = psnr_group_mean[i]
        psnr_mean = psnr_group_mean[i]

        distance_min = psnr_group_min[i]
        psnr_min = psnr_group_min[i]
        
        distance_max = psnr_group_max[i]
        psnr_max = psnr_group_max[i]
        

        test_save = save_group[i]
        d_score = score_group[i]

        distance_mean -= min(distance_mean)  # distance = (distance - min) / (max - min)
        distance_mean /= max(distance_mean)
        
        distance_min -= min(distance_min)  # distance = (distance - min) / (max - min)
        distance_min /= max(distance_min)
        
        distance_max -= min(distance_max)  # distance = (distance - min) / (max - min)
        distance_max /= max(distance_max)
        
        
        saves = np.concatenate((saves, test_save), axis=0)

        scores_mean = np.concatenate((scores_mean, distance_mean), axis=0)
        scores_min = np.concatenate((scores_min, distance_min), axis=0)
        scores_max = np.concatenate((scores_max, distance_max), axis=0)
        
        d_scores = np.concatenate((d_scores, d_score), axis=0)
        labels = np.concatenate((labels, gt[i][4:]), axis=0)  # Exclude the first 4 unpredictable frames in gt.
        psnr_source_mean = np.concatenate((psnr_source_mean, psnr_mean), axis=0)  # Exclude the first 4 unpredictable frames in gt.
        psnr_source_min = np.concatenate((psnr_source_min, psnr_min), axis=0)  # Exclude the first 4 unpredictable frames in gt.
        psnr_source_max = np.concatenate((psnr_source_max, psnr_max), axis=0)  # Exclude the first 4 unpredictable frames in gt.

    # print("scores: ",scores[:100])
    # print("labels: ",labels[:100])
    np.save("scores/score_discriminator_60", saves)
    np.save("scores/psnr_60_mean", psnr_source_mean)
    np.save("scores/psnr_60_min", psnr_source_min)
    np.save("scores/psnr_60_max", psnr_source_max)
    


    assert scores_mean.shape == labels.shape, \
        f'Ground truth has {labels.shape[0]} frames, but got {scores_mean.shape[0]} detected frames.'

    fpr_mean, tpr_mean, thresholds = metrics.roc_curve(labels, scores_mean, pos_label=0)
    fpr_min, tpr_min, thresholds = metrics.roc_curve(labels, scores_min, pos_label=0)
    fpr_max, tpr_max, thresholds = metrics.roc_curve(labels, scores_max, pos_label=0)
    
    fpr_s, tpr_s, thresholds_s = metrics.roc_curve(labels, d_scores, pos_label=0)

    auc_mean = metrics.auc(fpr_mean, tpr_mean)
    auc_max = metrics.auc(fpr_max, tpr_max)
    auc_min = metrics.auc(fpr_min, tpr_min)
    
    auc_s = metrics.auc(fpr_s, tpr_s)

    print(f'AUC: {auc_mean}\n')
    print(f'AUC: {auc_max}\n')
    print(f'AUC: {auc_min}\n')
    
    print(f'AUC_d: {auc_s}\n')
    
    return auc_mean


if __name__ == '__main__':
    args = parser.parse_args()
    test_cfg = update_config(args, mode='test')
    test_cfg.print_cfg()
    val(test_cfg)
    # Uncomment this to test the AUC mechanism.
    # labels = [0,  0,   0,   0,   0,  1,   1,    1,   0,  1,   0,    0]
    # scores = [0, 1/8, 2/8, 1/8, 1/8, 3/8, 6/8, 7/8, 5/8, 8/8, 2/8, 1/8]
    # fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    # print(fpr)
    # print('~~~~~~~~~~~~`')
    # print(tpr)
    # print('~~~~~~~~~~~~`')
    # print(thresholds)
    # print('~~~~~~~~~~~~`')