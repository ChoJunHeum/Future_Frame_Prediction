import scipy.io
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
video_folders = "/home/chojh21c/ADGW/Future_Frame_Prediction/datasets/avenue/ground_truth_demo/testing_label_mask/2_label.mat"
label_path = "/home/chojh21c/ADGW/Future_Frame_Prediction/datasets/avenue/avenue.mat"

mat = scipy.io.loadmat(video_folders)['volLabel']
labels = scipy.io.loadmat(label_path)

for i in range(1051, 1100):
    print(sum(sum(mat[0][i])))