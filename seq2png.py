# adapted from https://github.com/mitmul/caltech-pedestrian-dataset-converter/blob/master/scripts/convert_seqs.py

import os
import glob
import cv2 as cv


def save_img(save_dir, dname, fn, i, frame):
	print('{}/{}/{}_{}.png'.format(
		save_dir, os.path.basename(dname),
		os.path.basename(fn).split('.')[0], i))\

	cv.imwrite('{}/{}/{}_{}.png'.format(
		save_dir, os.path.basename(dname),
		os.path.basename(fn).split('.')[0], i), frame)

def convert(dir, save_dir):
	for dname in sorted(glob.glob(dir)):
		print(dname)
		for fn in sorted(glob.glob('{}/*.seq'.format(dname))):
			print(fn)
			cap = cv.VideoCapture(fn)
			i = 0
			while True:
				ret, frame = cap.read()
				if not ret:
					break
				save_img(save_dir, dname, fn, i, frame)
				i += 1
			print(fn)

convert('datasets/CalTech/training/*', 'datasets/CalTech/training')
convert('datasets/CalTech/testing/*', 'datasets/CalTech/testing')