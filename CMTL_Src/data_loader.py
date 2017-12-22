import os
import random
import sys
import numpy as np
import pandas as pd
import cv2

from Src.utils import show_density_map

DEBUG_SHOW = False

class ImageDataLoader:
    def __init__(self, data_path, gt_path, shuffle=False, num_classes=10, batch_size=10):
        self.data_path = data_path
        self.gt_path = gt_path
        self.data_files = [filename for filename in os.listdir(data_path)
                           if os.path.isfile(os.path.join(data_path, filename))]
        self.data_files.sort()
        self.shuffle = shuffle
        if shuffle:
            random.seed(2468)
        self.num_samples = len(self.data_files)
        self.blob_list = {}
        self.id_list = range(0, self.num_samples)
        #!!!!!!!!!!!!!!!!!!!!!!!!just for test!!!!!!!!!!!!!!!!!!!
        #self.id_list = range(0, 3)
        # !!!!!!!!!!!!!!!!!!!!!!!!just for test!!!!!!!!!!!!!!!!!!!
        self.batch_size = batch_size
        self.min_gt_count = sys.maxsize
        self.max_gt_count = 0
        self.num_classes = num_classes
        self.count_class_hist = np.zeros(self.num_classes)

        self.get_stats_in_dataset() #get min - max crowd count present in the dataset. used later for assigning crowd group/class

    def read_image_and_gt(self, fname):
        img = cv2.imread(self.data_path + '\\' + fname, 0)
        if img is None:
            print('error in loading image file: {}'.format(fname))
            return
        if DEBUG_SHOW: cv2.imshow('image', img)
        try:
            if img.shape[2] > 0:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except IndexError:
            pass

        wd, ht = img.shape
        # scale image to 1/4
        wd1, ht1 = wd // 2, ht // 2
        img = cv2.resize(img, (ht1, wd1))
        if DEBUG_SHOW: cv2.imshow('image resized', img)
        img = img.astype(np.float32, copy=False)
        img = img / 255.0
        img = img.reshape(1, img.shape[0], img.shape[1], 1)

        try:
            den = pd.read_csv(self.gt_path + '\\' + os.path.splitext(fname)[0] + '.csv', header=None,
                              dtype=np.float32).as_matrix()
            if DEBUG_SHOW: show_density_map('density map', den)
        except FileNotFoundError:
            print('error in loading density map of file: {}'.format(fname))
            return
        # print(np.sum(den))
        # for mcnn scale the input image to 1/4, the coresponding density map needs to scale to same size
        wd2, ht2 = wd1 // 4, ht1 // 4
        den = cv2.resize(den, (ht2, wd2), interpolation=cv2.INTER_CUBIC) * 4 * 16
        if DEBUG_SHOW: show_density_map('density map resized', den)
        print(np.sum(den))
        den = den.reshape(1, den.shape[0], den.shape[1], 1)
        gt_count = sum(den)
        print(np.sum(den))
        return img, den, gt_count

    def get_stats_in_dataset(self):
        min_count = sys.maxsize
        max_count = 0
        gt_count_array = np.zeros(self.num_samples)

        for i, fname in enumerate(self.data_files):
            try:
                den = pd.read_csv(self.gt_path + '\\' + os.path.splitext(fname)[0] + '.csv', header=None,
                                  dtype=np.float32).as_matrix()
            except FileNotFoundError:
                continue
            gt_count = np.sum(den)
            min_count = min(min_count, gt_count)
            max_count = max(max_count, gt_count)
            gt_count_array[i] = gt_count

        self.min_gt_count = min_count
        self.max_gt_count = max_count
        bin_val = (self.max_gt_count-self.min_gt_count)/float(self.num_classes)
        class_idx_array = np.round(gt_count_array/bin_val)

        for class_idx in class_idx_array:
            class_idx = int(min(class_idx, self.num_classes-1))
            self.count_class_hist[class_idx] += 1

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.data_files)
        files = self.data_files
        id_list = self.id_list

        num_batches = int(len(files)/self.batch_size)

        for i_batch in range(num_batches):
            blob = {'datas': [], 'gt_densitys': [], 'gt_class_labels': [], 'fnames': [], 'gt_counts': []}
            for idx in range(i_batch*self.batch_siz, (i_batch+1)*self.batch_siz):
                fname = files[idx]
                img, den, gt_count = self.read_image_and_gt(fname)
                if img is None or den is None:
                    continue
                gt_class_label = np.zeros(self.num_classes, dtype=np.int)
                bin_val = (self.max_gt_count - self.min_gt_count) / float(self.num_classes)
                class_idx = np.round(gt_count / bin_val)
                class_idx = int(min(class_idx, self.num_classes - 1))
                gt_class_label[class_idx] = 1
                blob['datas'].append(img)
                blob['gt_densitys'].append(den)
                blob['gt_class_labels'].append(gt_class_label.reshape(1, gt_class_label.shape[0]))
                blob['fnames'].append(fname)
                blob['gt_counts'].append(gt_count)
                # cv2.destroyAllWindows()

            blob['datas'] = np.concatenate(blob['datas'], 0)
            blob['gt_densitys'] = np.concatenate(blob['gt_densitys'], 0)
            blob['gt_class_labels'] = np.concatenate(blob['gt_class_labels'], 0)
            yield blob

    def get_num_samples(self):
        return self.num_samples

    def get_classifier_weights(self):
        wts = self.count_class_hist
        wts = 1 - wts/(sum(wts)*1.0)
        wts = wts /sum(wts)
        return wts

