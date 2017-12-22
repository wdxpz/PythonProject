import os
import random
import numpy as np
import pandas as pd
import cv2


class ImageDataLoader:
    def __init__(self, data_path, gt_path, shuffle=False, batch_size=10):
        self.data_path = data_path
        self.gt_path = gt_path
        self.batch_size = batch_size
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

    def __iter__(self):
        '''
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        the resize of density map will cause the difference with original density map, the best way is to
        use transpose conv2d in the network to resotore the gennerated density map with the same size as
        the input image
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        '''

        if self.shuffle:
                random.shuffle(self.data_files)
        files = self.data_files
        id_list = self.id_list

        num_batches = len(self.data_files) // self.batch_size
        for i_batch in range(num_batches):
            blob = {'data': [], 'gt_density': [], 'fname': []}
            for idx in range(i_batch*self.batch_size, (i_batch+1)*self.batch_size):
                fname = files[idx]
                img = cv2.imread(self.data_path + '\\' + fname, 0)
                if img is None:
                    continue
                # cv2.imshow('image', img)
                try:
                    if img.shape[2] > 0:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                except IndexError:
                    pass
                wd, ht = img.shape

                # scale image to 1/4
                wd1, ht1 = wd // 2, ht // 2
                img = cv2.resize(img, (ht1, wd1))
                img = img.astype(np.float32, copy=False)
                img = img / 255.0
                img = img.reshape(1, img.shape[0], img.shape[1], 1)
                try:
                    den = pd.read_csv(self.gt_path + '\\' + os.path.splitext(fname)[0] + '.csv', header=None,
                                      dtype=np.float32).as_matrix()
                except FileNotFoundError:
                    continue
                # print(np.sum(den))
                # for mcnn scale the input image to 1/4, the coresponding density map needs to scale to same size
                wd2, ht2 = wd1 // 4, ht1 // 4
                den = cv2.resize(den, (ht2, wd2), interpolation=cv2.INTER_CUBIC) * 4 * 16
                # print(np.sum(den))
                den = den.reshape(1, den.shape[0], den.shape[1], 1)
                blob['data'].append(img)
                blob['gt_density'].append(den)
                blob['fname'].append(fname)
                # cv2.destroyAllWindows()

            blob['data'] = np.concatenate(blob['data'], 0)
            blob['gt_density'] = np.concatenate(blob['gt_density'], 0)
            yield blob

    def get_num_samples(self):
        return self.num_samples
