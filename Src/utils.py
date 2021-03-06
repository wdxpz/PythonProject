import numpy as np
import skimage
from skimage import io
import os
import cv2


def save_results(input_img, gt_data, density_map, output_dir, fname='results.png'):
    input_img = input_img[0][0]
    gt_data = 255 * gt_data / np.max(gt_data)
    density_map = 255 * density_map / np.max(density_map)
    gt_data = gt_data[0][0]
    density_map = density_map[0][0]

    if density_map.shape[1] != input_img.shape[1]:
        density_map = skimage.transfomr.resize(density_map, input_img.shape[0], input_img.shape[1])
        gt_data = skimage.transfomr.resize(gt_data, input_img.shape[0], input_img.shape[1])

    result_img = np.hstack((input_img, gt_data, density_map))

    skimage.io.imsave(os.path.join(output_dir, fname), result_img)


def save_density_map(fname, den):
    den_int = den * 255 / np.max(den)
    den_int = den_int.astype(np.uint8, copy=False)
    cv2.imwrite(fname, den_int)


def show_density_map(name, den):
    den_int = den * 255 / np.max(den)
    den_int = den_int.astype(np.uint8, copy=False)
    cv2.imshow(name, den_int)


def display_results(input_img, gt_data, density_map):
    input_img = input_img[0][0]
    gt_data = 255 * gt_data / np.max(gt_data)
    density_map = 255 * density_map / np.max(density_map)
    gt_data = gt_data[0][0]
    density_map = density_map[0][0]
    if density_map.shape[1] != input_img.shape[1]:
        input_img = skimage.transfomr.resize(input_img, density_map.shape[0], density_map.shape[1])

    result_img = np.hstack((input_img, gt_data, density_map))
    result_img = result_img.astype(np.uint8, copy=False)

    skimage.imshow('Result', result_img)
