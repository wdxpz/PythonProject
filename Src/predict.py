import skimage
from skimage import io, transform
import tensorflow as tf
import numpy as np

output_dir = '../saved_models/'


def predict_crowd_count(filepathname):
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

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(output_dir + '.meta')
        loader.restore(sess, output_dir)

        # Get Tensors from loaded model
        loaded_input_img = loaded_graph.get_tensor_by_name('input_image:0')
        # loaded_leaning_rate = loaded_graph.get_tensor_by_name('learning_rate:0')
        loaded_is_trainging = loaded_graph.get_tensor_by_name('is_training:0')
        loaded_density_map = loaded_graph.get_tensor_by_name('density_map:0')

        density_map = sess.run(loaded_density_map, feed_dict={loaded_input_img: img, loaded_is_trainging: False})
        print("predict crowd count: {}".format(np.sum(density_map)))

if __name__ == '__main__':
    predict_crowd_count('test.jpg')