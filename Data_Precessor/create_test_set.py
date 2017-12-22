import os
import scipy.io as sio
import pandas as pd
import cv2

from Data_Precessor.get_density_map import get_density_map
from Src.utils import show_density_map

dataset = 'B'
dataset_name = 'part_'+dataset
path = '../Data/part_' + dataset + '_final/test_data/images/'
gt_path = '../Data/part_' + dataset + '_final/test_data/ground_truth/'
gt_path_csv = '../Data/part_' + dataset + '_final/test_data/ground_truth_csv/'

DEBUG_SHOW = False

if not os.path.exists(gt_path_csv):
    os.makedirs(gt_path_csv)

num_images = 182 if dataset == 'A' else 316

for i in range(1, num_images):
    if i % 10 ==0:
        print('processing {}/{}files...'.format(i, num_images))

    image_info = sio.loadmat(gt_path + 'GT_IMG_' + str(i) + '.mat')
    annPoints = image_info['image_info'][0][0][0][0]['location']

    input_img_name = path + 'IMG_' + str(i) + '.jpg'
    im = cv2.imread(input_img_name, 0)
    if im is None:
        continue
    if DEBUG_SHOW: cv2.imshow('image', im)
    [h, w] = im.shape

    # im_density = get_density_map(im, annPoints, model = 'constant_sigma')
    print('file {}'.format(input_img_name))
    im_density = get_density_map(im, annPoints)
    if DEBUG_SHOW:  show_density_map("density map", im_density)
    pd.DataFrame(im_density).to_csv(gt_path_csv + 'IMG_' + str(i) + '.csv', header=False, index=False)
    if DEBUG_SHOW:  cv2.destroyAllWindows()
