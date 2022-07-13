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
from util import psnr_error, psnr_error_target
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
parser.add_argument('--input_size', default=256, type=int, help='The img size.')


def val(cfg, model=None, dis_model=None, iters=None, mul_rate=0):
    if model:  # This is for testing during training.
        generator = model
        generator.eval()

        # discriminator = dis_model
        # discriminator.eval()

        cur_step = iters

    else:
        generator = vgg16bn_unet().cuda().eval()
        generator.load_state_dict(torch.load('weights/' + cfg.trained_model)['net_g'])
        
        # discriminator = PixelDiscriminator(input_nc=3).cuda().eval()
        # discriminator.load_state_dict(torch.load('weights/' + cfg.trained_model)['net_d'])

        cur_step = "test"


        print(f'The pre-trained generator has been loaded from \'weights/{cfg.trained_model}\'.\n')
        
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True).cuda()

    input_size = cfg.input_size
    video_folders = os.listdir(cfg.test_data)
    video_folders.sort()
    video_folders = [os.path.join(cfg.test_data, aa) for aa in video_folders]
    TT = ToTensor()

    fps = 0
    psnr_group_mean = []
    psnr_group_max = []
    psnr_group_min = []
    psnr_group = []
    targets = []

    save_group = []
    score_group = []

    gt_loader = Label_loader(cfg, video_folders)  # Get gt labels.
    gt = gt_loader()

    confidenc_score = 0.4
    bbsr = 0.3

    with torch.no_grad():
        for i, folder in enumerate(video_folders):
            label = gt[i][4:]

            dataset = Dataset.test_dataset(cfg, folder)

            psnrs_mean = []
            psnrs_max = []
            psnrs_min = []
            psnrs = []
            
            # dis_scores = []
            save_data = []
            for j, clip in enumerate(dataset):

                clip = torch.from_numpy(clip).unsqueeze(0)

                frame_1 = clip[:,0:3, :, :] # (n, 12, 256, 256) 
                frame_2 = clip[:,3:6, :, :] # (n, 12, 256, 256) 
                f_target = clip[:,6:9, :, :] # (n, 12, 256, 256) 
                frame_4 = clip[:,9:12, :, :] # (n, 12, 256, 256)
                frame_5 = clip[:,12:15, :, :]                
                 
                frame_1_crop = ((frame_1[0] + 1 ) / 2)[(2,1,0),...]
                frame_2_crop = ((frame_2[0] + 1 ) / 2)[(2,1,0),...]
                frame_4_crop = ((frame_4[0] + 1 ) / 2)[(2,1,0),...]
                frame_5_crop = ((frame_5[0] + 1 ) / 2)[(2,1,0),...]
                f_target_crop = ((f_target[0] + 1 ) / 2)[(2,1,0),...]   
                save_image(f_target_crop,f'crop_imgs/tester_g.png')

                img_1 = to_pil_image(frame_1_crop)
                img_2 = to_pil_image(frame_2_crop)
                img_4 = to_pil_image(frame_4_crop)
                img_5 = to_pil_image(frame_5_crop)
                img_t = to_pil_image(f_target_crop)
                # img_1.save(f'crop_imgs/tester_eval.png')

                results = yolo_model(img_4)
                areas = results.xyxy[0]
                
                new_areas = []
                patch_sizes = np.array([])

                for _, area in enumerate(areas):
                    
                    area = area.tolist()

                    if area[4] > confidenc_score and area[5] == 0:
                    # if area[5]==0:
                        xmin_ = area[0]
                        ymin_ = area[1]
                        xmax_ = area[2]
                        ymax_ = area[3]
                            
                        n_x = 2
                        n_y = 1.5

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
                        patch_sizes = np.append(patch_sizes, [x*y*0.1])
                targets.append(len(new_areas))
                if len(new_areas) != 0:

                    tframe_1 = torch.Tensor([])
                    tframe_2 = torch.Tensor([])
                    tframe_4 = torch.Tensor([])
                    tframe_5 = torch.Tensor([])
                    tframe_t = torch.Tensor([])
                    
                    for k, area in enumerate(new_areas):
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

                        # save_image(crop_img_1,f'crop_imgs/tester_1_{k}.png')
                        # save_image(crop_img_2,f'crop_imgs/tester_2_{k}.png')
                        # save_image(crop_img_3,f'crop_imgs/tester_3_{k}.png')
                        # save_image(crop_img_4,f'crop_imgs/tester_4_{k}.png')
                        # save_image(crop_img_t,f'crop_imgs/tester_t_{k}.png')

                    tframe_1 = torch.cat([tframe_1,frame_1 ],0)
                    tframe_2 = torch.cat([tframe_2,frame_2 ],0)
                    tframe_4 = torch.cat([tframe_4,frame_4 ],0)
                    tframe_5 = torch.cat([tframe_5,frame_5 ],0)
                    tframe_t = torch.cat([tframe_t,f_target],0)

                    frame_1 = tframe_1.cuda()
                    frame_2 = tframe_2.cuda()
                    frame_4 = tframe_4.cuda()
                    frame_5 = tframe_5.cuda()
                    f_target = tframe_t.cuda()

                    # print(frame_1.shape)

                    f_input = torch.cat([frame_1,frame_2, frame_4, frame_5], 1)
                    G_frame = generator(f_input)
                    
                    # save_image(G_frame,f'crop_imgs/tester_g.png')



                    psnr = psnr_error_target(G_frame, f_target).cpu().detach()
                    if len(new_areas) != 1:

                        patch_sizes -= min(patch_sizes)
                        patch_sizes /= max(patch_sizes)
                        patch_sizes = 1 + ((1-patch_sizes)*bbsr)
                        patch_sizes = np.append(patch_sizes, 1)

                        psnr = np.multiply(patch_sizes, psnr)
                    # print(psnr)
                    # quit()
                    # psnr: [target1_psnr, target2_psnr, ... , targetN_psnr]
                # else:
                    # d_score = torch.Tensor([])
                # print(f'psnr: {psnr} \ttarget:{len(new_areas)} \t Anomaly: {label[j]}')
                psnrs.append(psnr)
                psnrs_mean.append(torch.mean(psnr))
                psnrs_max.append(torch.max(psnr))
                psnrs_min.append(torch.min(psnr))
                # psnrs_*: target들 중 선택하여 추가
                # print("------------------------------\n")
                # print(psnr)
                # print(torch.max(psnr))
                # print(torch.min(psnr))
                # print("\n------------------------------\n")
                # save_data.append(d_score.cpu().detach().numpy())
                # dis_scores.append((torch.min(d_score)).cpu().detach().numpy())

                torch.cuda.synchronize()
                end = time.time()
                if j > 1:  # Compute fps by calculating the time used in one completed iteration, this is more accurate.
                    fps = 1 / (end - temp)
                temp = end
                # print(f'\rDetecting: [{i + 1:02d}] {j + 1}/{len(dataset)}, {fps:.2f} fps.', end='')
                print(f'\rDetecting: {j + 1}\t Anomaly: {label[j]} \tpsnr: {psnr}', end='\n')

            # break

            psnr_group.append(np.array(psnrs))
            psnr_group_mean.append(np.array(psnrs_mean))
            psnr_group_min.append(np.array(psnrs_min))
            psnr_group_max.append(np.array(psnrs_max))
            # psnr_group_*: 각 dataset의 psnrs_* 추가

            # score_group.append(np.array(dis_scores))

            save_group.append(save_data)

            if not model:
                if cfg.show_curve:
                    video_writer.release()
                    curve_writer.release()


    print('\nAll frames were detected, begin to compute AUC.')

    assert len(psnr_group_mean) == len(gt), f'Ground truth has {len(gt)} videos, but got {len(psnr_group)} detected videos.'

    scores_mean = np.array([], dtype=np.float32)
    scores_max = np.array([], dtype=np.float32)
    scores_min = np.array([], dtype=np.float32)
    
    labels = np.array([], dtype=np.int8)
    saves = np.array([], dtype=np.float32)
    # d_scores = np.array([], dtype=np.float32)
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
        # d_score = score_group[i]

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
        
        # d_scores = np.concatenate((d_scores, d_score), axis=0)
        labels = np.concatenate((labels, gt[i][4:]), axis=0)  
        psnr_source_mean = np.concatenate((psnr_source_mean, psnr_mean), axis=0)  
        psnr_source_min = np.concatenate((psnr_source_min, psnr_min), axis=0)  
        psnr_source_max = np.concatenate((psnr_source_max, psnr_max), axis=0)  

    # print("scores: ",scores[:100])
    # print("labels: ",labels[:100])
    # np.save(f"scores/score_discriminator_resize_256_{cur_step}", saves)
    np.save(f"scores/psnr_resize_256_{cur_step}_mean", psnr_group_mean)
    np.save(f"scores/psnr_resize_256_{cur_step}_min", psnr_group_min)
    np.save(f"scores/psnr_resize_256_{cur_step}_max", psnr_group_max)
    np.save(f"scores/psnr_resize_256_{cur_step}", psnr_group)
    np.save(f"scores/target_count", targets)
    
    


    assert scores_mean.shape == labels.shape, \
        f'Ground truth has {labels.shape[0]} frames, but got {scores_mean.shape[0]} detected frames.'


    fpr_mean, tpr_mean, thresholds = metrics.roc_curve(labels, scores_mean, pos_label=0)
    fpr_min, tpr_min, thresholds = metrics.roc_curve(labels, scores_min, pos_label=0)
    fpr_max, tpr_max, thresholds = metrics.roc_curve(labels, scores_max, pos_label=0)
    
    # fpr_s, tpr_s, thresholds_s = metrics.roc_curve(labels, d_scores, pos_label=0)

    auc_mean = metrics.auc(fpr_mean, tpr_mean)
    auc_max = metrics.auc(fpr_max, tpr_max)
    auc_min = metrics.auc(fpr_min, tpr_min)
    
    # auc_s = metrics.auc(fpr_s, tpr_s)
    res = open("res.txt", "w")
    print(f'AUC_mean: {auc_mean}\n')
    print(f'AUC_max: {auc_max}\n')
    print(f'AUC_min: {auc_min}\n')

    res.write(f"mul_rate: {mul_rate}\n")
    res.write(f"auc_min: {auc_min}\n")
    res.write(f"auc_mean: {auc_mean}\n\n")

    res.close()

    print(f"Trained_model: {cfg.trained_model} | confidence score: {confidenc_score} | bbsr: {bbsr}")

    
    # print(f'AUC_d: {auc_s}\n')
    
    return auc_mean


if __name__ == '__main__':
    args = parser.parse_args()
    test_cfg = update_config(args, mode='test')
    test_cfg.print_cfg()
    mul_rates = [x for x in range(5,20)]

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