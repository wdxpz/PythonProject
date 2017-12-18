from Src.data_loader import ImageDataLoader
from Src.model import *
from math import fabs
import skimage
from skimage import io, transform
import numpy as np
import sys

dataset = 'B'
output_dir = '../saved_models/'
train_path = '../data/formatted_trainval/part_' + dataset + '_patches_9/train'
train_gt_path = '../data/formatted_trainval/part_' + dataset + '_patches_9/train_den'
val_path = '../data/formatted_trainval/part_' + dataset + '_patches_9/val'
val_gt_path = '../data/formatted_trainval/part_' + dataset + '_patches_9/val_den'
test_path = '../data/part_' + dataset + '_final/test_data/images'
test_gt_path = '../data/part_' + dataset + '_final/test_data/ground_truth_csv'

data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=True, pre_load=False)
data_loader_val = ImageDataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=True, pre_load=False)
data_loader_test = ImageDataLoader(test_path, test_gt_path, shuffle=False, gt_downsample=True, pre_load=False)


def evaluate(sess, dataloader, input_image, gt_density, loss, gt_count, crowd_count, is_train):
    val_loss = 0
    mae = 0
    steps = 0
    max_err = 0
    min_err = sys.maxsize
    for blob in dataloader:
        steps += 1
        im_data = blob['data']
        gt_data = blob['gt_density']
        loss_val, gt_count_val, crowd_count_val = sess.run([loss, gt_count, crowd_count],
                                                           feed_dict={input_image: im_data, gt_density: gt_data,
                                                                      is_train: False})
        val_loss += loss_val
        err = fabs(gt_count_val - crowd_count_val)
        mae += err
        if err > max_err: max_err = err
        if err < min_err: min_err = err
        # print('validating file: {}, ground truth count: {}, estimated count: {}'.format(blob['fname'],
        #                                                                                 gt_count_val, crowd_count_val))
    return val_loss / steps, mae / steps, max_err, min_err


def test(sess, dataloader, input_image, gt_density, loss, gt_count, crowd_count, is_train):
    val_loss = 0
    mae = 0
    steps = 0
    max_err = 0
    min_err = sys.maxsize
    for blob in dataloader:
        steps += 1
        im_data = blob['data']
        gt_data = blob['gt_density']
        loss_val, gt_count_val, crowd_count_val = sess.run([loss, gt_count, crowd_count],
                                                           feed_dict={input_image: im_data, gt_density: gt_data,
                                                                      is_train: False})
        val_loss += loss_val
        err = fabs(gt_count_val - crowd_count_val)
        mae += err
        if err > max_err: max_err = err
        if err < min_err: min_err = err
        # print('testing file: {}, ground truth count: {}, estimated count: {}'.format(blob['fname'],
        #                                                                              gt_count_val,crowd_count_val))
    return val_loss / steps, mae / steps, max_err, min_err


def predict(sess, filepathname, density, input_image, is_train):
    try:
        img = skimage.io.imread(filepathname)
    except:
        return
    try:
        if img.shape[2] > 0:
            img = skimage.color.rgb2gray(img)
    except IndexError:
        pass
    wd, ht = img.shape

    wd1, ht1 = (wd // 4) * 4, (ht // 4) * 4
    img = skimage.transform.resize(img, [wd1, ht1], mode='constant')
    skimage.io.imshow(img)
    img = img.reshape(1, img.shape[0], img.shape[1], 1)

    density_map = sess.run(density, feed_dict={input_image: img, is_train: False})
    print("predict crowd count: {}".format(np.sum(density_map)))


def train(epoch_count, learning_rate, beta1, dropout, bn):
    print_every = 500
    valid_every = 1000

    input_image, gt_density, lr, is_train = model_input()
    density = model_MCNN(input_image, bn, dropout, is_train)
    density = tf.identity(density, name='density_map')
    loss, gt_count, crowd_count, re_count = model_loss(gt_density, density)
    opt = model_opt(loss, learning_rate, beta1, bn)
    # gt_count, crowd_count = model_crowd_count(gt_density, density)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(1, epoch_count+1):
            steps = 0
            train_loss = 0
            mae = 0
            for blob in data_loader:
                steps += 1
                im_data = blob['data']
                # imshow(np.squeeze(im_data))
                gt_data = blob['gt_density']
                # print(np.sum(gt_data))

                _, loss_val, gt_count_val, crowd_count_val, re_count_val = \
                    sess.run([opt, loss, gt_count, crowd_count, re_count],
                                                                      feed_dict={input_image: im_data,
                                                                                 gt_density: gt_data,
                                                                                 lr: learning_rate,
                                                                                 is_train: True})
                train_loss += loss_val
                mae += fabs(gt_count_val-crowd_count_val)

                if steps % print_every == 0:
                    log_text = 'epoch: %4d, step %4d, avaerage training loss: %f, average training mae: %4.1f,  ' % \
                               (epoch_i, steps, train_loss / steps,  mae / steps)
                    print(log_text)

                if steps % valid_every == 0:
                    avg_val_loss, val_mae, max_err, min_err = evaluate(sess, data_loader_val, input_image, gt_density,
                                                     loss, gt_count, crowd_count, is_train)

                    log_text = 'epoch: %4d, step %4d, average training loss: %f, average training mae: %4.1f, ' \
                               'average validation loss: %f, average validation mae: %4.1f, max validation err: ' \
                               '%4.1f, min validation err: %4.1f' % \
                               (epoch_i, steps, train_loss / steps, mae / steps, avg_val_loss, val_mae, max_err, min_err)
                    print(log_text)

                    saver.save(sess, output_dir)

            if epoch_i % 5 == 0:
                # final evaluation on test set
                avg_test_loss, test_mae, max_err, min_err = test(sess, data_loader_test, input_image, gt_density,
                                                                 loss, gt_count, crowd_count, is_train)
                log_text = 'average test loss: %f, average test mae: %4.1f, , max test err: %4.1f, min test err: %4.1f' \
                           % (avg_test_loss, test_mae, max_err, min_err)
                print(log_text)

        predict(sess, 'test.jpg', density, input_image, is_train)

        saver.save(sess, output_dir)


if __name__ == '__main__':
    epoch_count = 100
    learning_rate = 0.0001
    beta1 = 0.5
    bn = False
    dropout = True
    train(epoch_count, learning_rate, beta1,dropout, bn)
