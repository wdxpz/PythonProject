import random
import os
import math
import cv2
import scipy.io as sio
import numpy as np
import pandas as pd
from Data_Precessor.get_density_map import get_density_map
from Src.utils import show_density_map, save_density_map

seed = 95461354
random.seed(seed)
N = 9
dataset = 'B'
dataset_name = 'part_' + dataset + '_patches_' + str(N)
path = '../data/part_' + dataset + '_final/train_data/images/'
gt_path = '../data/part_' + dataset + '_final/train_data/ground_truth/'
output_path = '../data/formatted_trainval/' + dataset_name
train_path_img = output_path + '/train/'
train_path_den = output_path + '/train_den/'
val_path_img = output_path + '/val/'
val_path_den = output_path + '/val_den/'

DEBUG_SHOW = False


if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(train_path_img):
    os.mkdir(train_path_img)
if not os.path.exists(train_path_den):
    os.mkdir(train_path_den)
if not os.path.exists(val_path_img):
    os.mkdir(val_path_img)
if not os.path.exists(val_path_den):
    os.mkdir(val_path_den)

num_images = 300 if dataset == 'A' else 400
num_val = math.ceil(num_images * 0.1)
indices = list(range(1, num_images+1))
random.shuffle(indices)

for i, idx in enumerate(indices):
    # if (i + 1) % 10 == 0:
    print('*'*20)
    print('processing {}th / {} files'.format(i + 1, num_images) + path + 'IMG_' + str(idx) + '.jpg')
    image_info = sio.loadmat(gt_path + 'GT_IMG_' + str(idx) + '.mat')
    annPoints = image_info['image_info'][0][0][0][0]['location']
    # print(annPoints.shape)
    input_img_name = path + 'IMG_' + str(idx) + '.jpg'
    im = cv2.imread(input_img_name, 0)
    if im is None:
        continue
    if DEBUG_SHOW:
        cv2.imshow('original image', im)
    [h, w] = im.shape

    # setting the sample area size, it is better not less than width/4, height/4
    sample_factor = 8
    if sample_factor <= 2:
        raise Exception('Error of sample_factor!')

    wn2 = sample_factor * math.floor(w / (sample_factor * sample_factor))
    hn2 = sample_factor * math.floor(h / (sample_factor * sample_factor))

    if w <= 2 * wn2:
        print('in case of the width size of sample area is less than 0')
        im = cv2.resize(im, (2 * wn2 + 1, h))
        if DEBUG_SHOW: cv2.imshow('image resize 1', im)
        annPoints[:, 0] = annPoints[:, 0] * 2 * wn2 / w
    if h <= 2 * hn2:
        print('in case of the height size of sample area is less than 0')
        im = cv2.resize(im, (w, 2 * hn2 + 1))
        if DEBUG_SHOW: cv2.imshow('image resize 2', im)
        annPoints[:, 1] = annPoints[:, 1] * 2 * hn2 / h
    [h, w] = im.shape

    im_density = get_density_map(im, annPoints)
    if DEBUG_SHOW: show_density_map('full density map', im_density)

    # # setting the sampling area of top left point of sample window
    # a_w = wn2 + 1
    # b_w = w - wn2
    # a_h = hn2 + 1
    # b_h = h - hn2
    #
    # for j in range(N):
    #     ##setting the top left point of sample window
    #     x = math.floor((b_w - a_w) * random.random() + a_w)
    #     y = math.floor((b_h - a_h) * random.random() + a_h)
    #     x1, x2 = x - wn2, x + wn2
    #     y1, y2 = y - hn2, y + hn2
    #     im_sampled = im[y1:y2, x1:x2]
    #     im_density_sampled = im_density[y1:y2, x1:x2]
    #     if DEBUG_SHOW: cv2.imshow('sample image {}'.format(j), im_sampled)
    #     if DEBUG_SHOW: show_density_map('sample density map {}'.format(j), im_density_sampled)
    #
    #     annPoints_sampled_count = 0
    #     for point in annPoints:
    #         if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
    #             annPoints_sampled_count += 1
    #     im_density_sampled_count = np.sum(im_density_sampled)
    #     print('density count in {}th sampled original ground true {}; in sampled density map {}'.
    #             format(j + 1, annPoints_sampled_count, im_density_sampled_count))
    #
    #     img_idx = str(idx) + '_' + str(j + 1)
    #     if i < num_val:
    #         cv2.imwrite(val_path_img + img_idx + '.jpg', im_sampled)
    #         pd.DataFrame(im_density_sampled).to_csv(val_path_den + img_idx + '.csv', header=False, index=False)
    #
    #         #to validate the save sample density map
    #         # den = pd.read_csv(val_path_den + img_idx + '.csv', header=None, dtype=np.float32).as_matrix()
    #         # print('density count in {}th saved sample density map {}'.
    #         #       format(j + 1,  np.sum(den)))
    #     else:
    #         cv2.imwrite(train_path_img + img_idx + '.jpg', im_sampled)
    #         pd.DataFrame(im_density_sampled).to_csv(train_path_den + img_idx + '.csv', header=False, index=False)

    img_idx = str(idx) + '_' + str(1)
    if i < num_val:
        cv2.imwrite(val_path_img + img_idx + '.jpg', im)
        save_density_map(val_path_den + img_idx + '_density.bmp', im_density)
        pd.DataFrame(im_density).to_csv(val_path_den + img_idx + '.csv', header=False, index=False)

        # to validate the save sample density map
        # den = pd.read_csv(val_path_den + img_idx + '.csv', header=None, dtype=np.float32).as_matrix()
        # print('density count in {}th saved sample density map {}'.
        #       format(j + 1,  np.sum(den)))
    else:
        cv2.imwrite(train_path_img + img_idx + '.jpg', im)
        save_density_map(train_path_den + img_idx + '_density.bmp', im_density)
        pd.DataFrame(im_density).to_csv(train_path_den + img_idx + '.csv', header=False, index=False)

