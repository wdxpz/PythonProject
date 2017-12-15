from Src.data_loader import ImageDataLoader
from Src.model import *
from math import fabs
from skimage.io import imshow
import numpy as np

dataset = 'A'
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
    for blob in dataloader:
        steps += 1
        im_data = blob['data']
        gt_data = blob['gt_density']
        loss_val, gt_count_val, crowd_count_val = sess.run([loss, gt_count, crowd_count],
                                                           feed_dict={input_image: im_data, gt_density: gt_data,
                                                                      is_train: False})
        val_loss += loss_val
        mae += fabs(gt_count_val - crowd_count_val)
    return val_loss / steps, mae / steps


def test(sess, dataloader, input_image, gt_density, loss, gt_count, crowd_count, is_train):
    val_loss = 0
    mae = 0
    steps = 0
    for blob in dataloader:
        steps += 1
        im_data = blob['data']
        gt_data = blob['gt_density']
        loss_val, gt_count_val, crowd_count_val = sess.run([loss, gt_count, crowd_count],
                                                           feed_dict={input_image: im_data, gt_density: gt_data,
                                                                      is_train: False})
        val_loss += loss_val
        mae += fabs(gt_count_val - crowd_count_val)
    return val_loss / steps, mae / steps


def train(epoch_count, learning_rate, beta1, dropout, bn):
    print_every = 100
    valid_every = 200

    input_image, gt_density, lr, is_train = model_input()
    density = model_MCNN(input_image, bn, dropout, is_train)
    density = tf.identity(density, name='density_map')
    loss, gt_count, crowd_count, re_count = model_loss(gt_density, density)
    opt = model_opt(loss, learning_rate, beta1, bn)
    # gt_count, crowd_count = model_crowd_count(gt_density, density)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(1, epoch_count):
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
                mae = fabs(gt_count_val-crowd_count_val)

                if steps % print_every == 0:
                    log_text = 'epoch: %4d, step %4d, avaerage training loss: %f, average training mae: %4.1f,  ' % \
                               (epoch_i, steps, train_loss / steps,  mae / steps)
                    print(log_text)

                if steps % valid_every == 0:
                    avg_val_loss, val_mae = evaluate(sess, data_loader_val, input_image, gt_density,
                                                     loss, gt_count, crowd_count, is_train)

                    log_text = 'epoch: %4d, step %4d, average training loss: %f, average training mae: %4.1f, ' \
                               'average validation loss: %f, average validation mae: %4.1f' % \
                               (epoch_i, steps, train_loss / steps, mae / steps, avg_val_loss, val_mae)
                    print(log_text)

        # final evaluation on test set
        avg_test_loss, test_mae = test(sess, data_loader_test, input_image, gt_density,
                                 loss, gt_count, crowd_count, is_train)
        log_text = 'epoch: %4d, step %4d, average test loss: %f, average test mae: %4.1f, ' % (avg_test_loss, test_mae)
        print(log_text)

        saver.save(sess, output_dir)


if __name__ == '__main__':
    epoch_count = 10
    learning_rate = 0.00005
    beta1 = 0.5
    bn = True
    dropout = True
    train(epoch_count, learning_rate, beta1,dropout, bn)
