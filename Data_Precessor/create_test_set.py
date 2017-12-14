import random
import os
import math
import skimage
from skimage import io
import scipy.io as sio
import pandas as pd

from Data_Precessor.get_density_map import get_density_map

dataset = 'A';
dataset_name = 'part_'+dataset
path = '../Data/part_' + dataset + '_final/test_data/images/'
gt_path = '../Data/part_' + dataset + '_final/test_data/ground_truth/'
gt_path_csv = '../Data/part_' + dataset + '_final/test_data/ground_truth_csv/'

if not os.path.exists(gt_path_csv):
    os.makedirs(gt_path_csv)

num_images = 182 if dataset == 'A' else 316

for i in range(1, num_images):
    if i % 10 ==0:
        print('processing {}/{}files...'.format(i, num_images))

    image_info = sio.loadmat(gt_path + 'GT_IMG_' + str(i) + '.mat')
    annPoints = image_info['image_info'][0][0][0][0]['location']

    input_img_name = path + 'IMG_' + str(i) + '.jpg'
    im = skimage.io.imread(input_img_name)
    im = skimage.color.rgb2gray(im)
    # skimage.io.imshow(im)

    im_density = get_density_map(im, annPoints)
    pd.DataFrame(im_density).to_csv(gt_path_csv + 'IMG_' + str(i) + '.csv', header=False, index=False)
