import cv2
import math
import numpy as np
import os.path
from Src.data_loader import ImageDataLoader
from Src.model import *

dataset = 'B'
output_dir = '../saved_models/'
train_path = '../data/formatted_trainval/part_' + dataset + '_patches_9/train'
train_gt_path = '../data/formatted_trainval/part_' + dataset + '_patches_9/train_den'
val_path = '../data/formatted_trainval/part_' + dataset + '_patches_9/val'
val_gt_path = '../data/formatted_trainval/part_' + dataset + '_patches_9/val_den'
test_path = '../data/part_' + dataset + '_final/test_data/images'
test_gt_path = '../data/part_' + dataset + '_final/test_data/ground_truth_csv'

BATCH_SIZE = 20

data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, batch_size=BATCH_SIZE)
data_loader_val = ImageDataLoader(val_path, val_gt_path, shuffle=False, batch_size=BATCH_SIZE)
data_loader_test = ImageDataLoader(test_path, test_gt_path, shuffle=False, batch_size=BATCH_SIZE)


def evaluate(sess, dataloader, input_image, gt_density, density, mae, mse):
    mae = 0
    mse = 0
    loss = 0
    num_samples = 0
    for blob in dataloader:
        num_samples += len(blob['fname'])
        im_data = blob['data']
        gt_data = blob['gt_density']

        density_map = sess.run(density, feed_dict={input_image: im_data, gt_density: gt_data})

        mae += np.sum(np.abs(np.sum(blob['gt_density'], (1, 2)) - np.sum(density_map, (1, 2))))
        mse += np.sum(np.square(np.sum(blob['gt_density'], (1,2))- np.sum(density_map, (1,2))))
        loss += np.sum(np.sqrt(np.sum(np.square(blob['gt_density']-density_map), (1, 2))))

    mae = mae / num_samples
    mse = math.sqrt(mse / num_samples)
    loss = loss / num_samples

    return loss, mse, mae


# def test(sess, dataloader, input_image, gt_density, loss, gt_count, crowd_count, is_train):
#     val_loss = 0
#     mae = 0
#     steps = 0
#     max_err = 0
#     min_err = sys.maxsize
#     max_file = ''
#     max_err_gt = 0
#     for blob in dataloader:
#         steps += 1
#         im_data = blob['data']
#         gt_data = blob['gt_density']
#         loss_val, gt_count_val, crowd_count_val = sess.run([loss, gt_count, crowd_count],
#                                                            feed_dict={input_image: im_data, gt_density: gt_data,
#                                                                       is_train: False})
#         val_loss += loss_val
#         err = fabs(gt_count_val - crowd_count_val)
#         mae += err
#
#         if err > max_err:
#             max_err = err
#             max_file = blob['fname']
#             max_err_gt = gt_count_val
#         if err < min_err: min_err = err
#     print('max test err file: {}, ground truth count: {}, estimated count: {}'.format(max_file,
#                                                                                       max_err_gt, crowd_count_val))
#     return val_loss / steps, mae / steps, max_err, min_err


def predict(sess, filepathname, density, input_image):
    img = cv2.imread(filepathname, 0)
    if img is None: return
    img = img.reshape(1, img.shape[0], img.shape[1], 1)
    density_map = sess.run(density, feed_dict={input_image: img})
    cv2.imshow('{} -- {}'.format(filepathname, np.sum(density_map)), np.squeeze(img))
    cv2.waitKey()


def train(epoch_count, beta1, dropout):
    input_image, gt_density, lr = model_input(BATCH_SIZE)
    density = model_MCNN(input_image, dropout)
    density = tf.identity(density, name='density_map')
    loss, mae, mse, = model_loss(gt_density, density)

    global_step = tf.Variable(0, dtype=tf.float32)
    learningrate = tf.train.exponential_decay(learning_rate=0.000001,global_step= global_step,
                                              decay_steps=500,
                                              decay_rate=0.8,
                                              staircase=True)
    # opt = tf.train.MomentumOptimizer(learningrate, momentum=0.9).minimize(loss, global_step=global_step)
    opt = tf.train.AdamOptimizer(learning_rate=learningrate, beta1=0.9).minimize(loss, global_step=global_step)
    saver = tf.train.Saver()
    tf.add_to_collection('train_opt', opt)

    # training_loss_sum = tf.summary.scalar('training_loss', loss)
    # validation_loss_sum = tf.summary.scalar('validation_loss', loss)

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(output_dir, sess.graph)

        sess.run(tf.global_variables_initializer())

        for epoch_i in range(1, epoch_count+1):
            # use total batches to train opt at first, then evaluate
            batch_steps = 0
            for blob in data_loader:
                batch_steps += 1
                im_data = blob['data']
                gt_data = blob['gt_density']

                _, density_map, t_loss, t_mae, t_mse, num_global_step, t_learningrate = sess.run([opt, density, loss, mae, mse, global_step, learningrate],
                                                   feed_dict={input_image: im_data, gt_density: gt_data})
                # loss = np.mean(np.sqrt(np.sum(np.square(blob['gt_density'] - density_map), (1, 2))))
                # np_mae = np.mean(np.abs(np.sum(blob['gt_density'], (1,2))- np.sum(density_map, (1,2))))
                # np_mse = np.sqrt(np.mean(np.square(np.sum(blob['gt_density'], (1,2))- np.sum(density_map, (1,2)))))

                # for i in range(BATCH_SIZE):
                #     print('file: {}, orginal count: {}, estimate count:{}'.format(blob['fname'][i],
                #                                                                   np.sum(blob['gt_density'][i]),
                #                                                                   np.sum(density_map[i])))
                print('Train -- epoch: {}, \tbatch: {}, \tglobal_step: {}\tlearningrate: {}\tloss: {}, \tmse: {}, \tmae: {}'.
                      format(epoch_i, batch_steps, num_global_step, t_learningrate, t_loss, t_mse, t_mae))
                # break

                saver.save(sess, output_dir)

            v_loss, v_mse, v_mae = evaluate(sess,data_loader_val, input_image, gt_density, density, mae, mse)
            print('-'*100)
            print('Validation -- epoch: {}, \tloss: {}, \tmse: {}, \tmae: {}'.format(epoch_i, v_loss, v_mse, v_mae))
            print('-' * 100)
            saver.save(sess, output_dir)
            # break
        predict(sess, 'test.jpg', density, input_image)


def re_train(epoch_count, beta1, dropout):
    if not os.path.exists(output_dir + '.meta'):
        return

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(output_dir + '.meta')
        loader.restore(sess, output_dir)

        # Get Tensors from loaded model
        input_image = loaded_graph.get_tensor_by_name('input_image:0')
        gt_density = loaded_graph.get_tensor_by_name('ground_true:0')
        density = loaded_graph.get_tensor_by_name('density_map:0')
        loss = loaded_graph.get_tensor_by_name('Mean_2:0')
        mae = loaded_graph.get_tensor_by_name('Mean:0')
        mse = loaded_graph.get_tensor_by_name('Sqrt:0')
        learningrate = loaded_graph.get_tensor_by_name('ExponentialDecay:0')
        opt = tf.get_collection('train_opt')[0]

        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(output_dir, sess.graph)

        # sess.run(tf.global_variables_initializer())
        global_step = tf.Variable(0, dtype=tf.float32)
        sess.run(tf.variables_initializer([global_step]))
        learningrate = tf.train.exponential_decay(learning_rate=0.0000001, global_step=global_step,
                                                  decay_steps=1000,
                                                  decay_rate=0.8,
                                                  staircase=True)

        for epoch_i in range(1, epoch_count + 1):
            # use total batches to train opt at first, then evaluate
            batch_steps = 0
            for blob in data_loader:
                batch_steps += 1
                im_data = blob['data']
                gt_data = blob['gt_density']

                _, density_map, t_loss, t_mae, t_mse, num_global_step, t_learningrate = sess.run(
                    [opt, density, loss, mae, mse, global_step, learningrate],
                    feed_dict={input_image: im_data, gt_density: gt_data})
                print('Train -- epoch: {}, \tbatch: {}, \tglobal_step: {}\tlearningrate: {}\tloss: {}, \tmse: {}, \tmae: {}'.
                    format(epoch_i, batch_steps, num_global_step, t_learningrate, t_loss, t_mse, t_mae))

            v_loss, v_mse, v_mae = evaluate(sess, data_loader_val, input_image, gt_density, density, mae, mse)
            saver.save(sess, output_dir)
            print('-' * 100)
            print('Validation -- epoch: {}, \tloss: {}, \tmse: {}, \tmae: {}'.format(epoch_i, v_loss, v_mse, v_mae))
            print('-' * 100)

            # break
        predict(sess, 'test.jpg', density, input_image)

if __name__ == '__main__':
    epoch_count = 1000
    learning_rate = 0.00001
    beta1 = 0.5
    bn = False
    dropout = False
    train(epoch_count, beta1,dropout)
    # re_train(epoch_count, beta1, dropout)
