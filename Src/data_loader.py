import os
import random

import numpy as np
import skimage
from skimage import io, transform
import pandas as pd
import cv2


class ImageDataLoader:
    def __init__(self, data_path, gt_path, shuffle=False, gt_downsample=False, pre_load=False):
        self.data_path = data_path
        self.gt_path = gt_path
        self.gt_downsample = gt_downsample
        self.pre_load = pre_load
        self.data_files = [filename for filename in os.listdir(data_path)
                           if os.path.isfile(os.path.join(data_path, filename))]
        self.data_files.sort()
        self.shuffle = shuffle
        if shuffle:
            random.seed(2468)
        self.num_samples = len(self.data_files)
        self.blob_list = {}
        #!!!!!!!!!!!!!!!!!!!!!!!!just for test!!!!!!!!!!!!!!!!!!!
        #self.id_list = range(0, 3)
        # !!!!!!!!!!!!!!!!!!!!!!!!just for test!!!!!!!!!!!!!!!!!!!
        self.id_list = range(0, self.num_samples)
        # if self.pre_load:
        #     print('Pre_loading the data. This may take some time...')
        #     for idx, fname in enumerate(self.data_files):
        #         # fn = os.path.join(self.data_path, fname)
        #         # img = skimage.io.imread(self.data_path+'\\'+fname)
        #         img = cv2.imread(self.data_path+'\\'+fname, 0)
        #         cv2.imshow('image', img)
        #         img = img.astype(np.float32, copy=False)
        #         try:
        #             if img.shape[2] > 0:
        #                 # img = skimage.color.rgb2gray(img)
        #                 img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #                 cv2.imshow('image gray', img)
        #         except IndexError:
        #             pass
        #         wd, ht = img.shape
        #         wd1, ht1 = (wd // 4) * 4, (ht // 4) * 4
        #         # img = skimage.transform.resize(img, [wd1, ht1], mode='constant')
        #         img = cv2.resize(img, (wd1, ht1))
        #         cv2.imshow('image resized', img)
        #         # print(np.max(img))
        #         # print(np.min(img))
        #         # skimage.io.imshow(img)
        #         img = img.reshape(1, img.shape[0], img.shape[1], 1)
        #         den = pd.read_csv(self.gt_path + '\\' + os.path.splitext(fname)[0] + '.csv', header=None, dtype=np.float32) \
        #             .as_matrix()
        #         # print(np.sum(den))
        #         #!!!!!!!!!!!!!!!!!!!!!!!!!
        #         #这部分downsample的计算移入到神经网络图的loss计算中，需验证！
        #         #!!!!!!!!!!!!!!!!!!!!!!!!!
        #         # if self.gt_downsample:
        #         #     wd1, ht1 = wd1/4, ht1/4
        #         # den = skimage.transform.resize(den, [wd1, ht1], mode='constant') * ((wd * ht) / (wd1 * ht1))
        #         den = cv2.resize(den, (wd1, ht1))
        #         cv2.imshow('density map', den)
        #         # print(np.sum(den))
        #         den = den.reshape(1, den.shape[0], den.shape[1], 1)
        #         blob = {'data': img, 'gt_density': den, 'fname': fname}
        #         self.blob_list[idx] = blob
        #         if (idx + 1) % 100 == 0:
        #             print('Loaded', idx + 1, '/', self.num_samples, 'files')
        #     print('complete loading', idx + 1, 'files')

    def __iter__(self):
        if self.shuffle:
            if self.pre_load:
                random.shuffle(self.id_list)
            else:
                random.shuffle(self.data_files)
        files = self.data_files
        id_list = self.id_list

        for idx in id_list:
            if self.pre_load:
                blob = self.blob_list[idx]
            else:
                fname = files[idx]
                # fname = '133_1.jpg'
                # img = skimage.io.imread(self.data_path + '\\' + fname)
                img = cv2.imread(self.data_path + '\\' + fname, 0)
                if img is None:
                    continue
                # cv2.imshow('image', img)

                try:
                    if img.shape[2] > 0:
                        # img = skimage.color.rgb2gray(img)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        # cv2.imshow('image gray', img)
                except IndexError:
                    pass
                wd, ht = img.shape

                wd1, ht1 = (wd // 4) * 4, (ht // 4) * 4
                # img = skimage.transform.resize(img, [wd1, ht1], mode='constant')
                img = cv2.resize(img, (ht1, wd1))
                # cv2.imshow('image resized', img)
                img = img.astype(np.float32, copy=False)
                img = img / 255.0
                img = img.reshape(1, img.shape[0], img.shape[1], 1)
                try:
                    den = pd.read_csv(self.gt_path + '\\' + os.path.splitext(fname)[0] + '.csv', header=None,
                                      dtype=np.float32).as_matrix()
                    # den_int = den * 255 / np.max(den)
                    # den_int = den_int.astype(np.uint8, copy=False)
                    # cv2.imshow('density map', den_int)
                except FileNotFoundError:
                    continue
                # print(fname)
                # print(np.max(den))
                # print(np.min(den))
                # den = skimage.transform.resize(den, [wd1, ht1], mode='constant') * ((wd * ht) / (wd1 * ht1))
                den = cv2.resize(den, (ht1, wd1))
                # den_int = den * 255 / np.max(den)
                # den_int = den_int.astype(np.uint8, copy=False)
                # cv2.imshow('density map resized', den_int)
                den = den.reshape(1, den.shape[0], den.shape[1], 1)
                # print(np.sum(den))
                blob = {'data': img, 'gt_density': den, 'fname': fname}

                # cv2.destroyAllWindows()

            yield blob

    def get_num_samples(self):
        return self.num_samples
