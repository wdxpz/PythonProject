import tensorflow as tf
import numpy as np
import cv2

output_dir = '../saved_models/'


def predict_crowd_count(filepathname):
    img = cv2.imread(filepathname, 0)
    if img is None:
        return
    try:
        if img.shape[2] > 0:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except IndexError:
        pass
    wd, ht = img.shape

    wd1, ht1 = wd // 2, ht // 2
    img = cv2.resize(img, (ht1, wd1))
    img_org = np.copy(img)
    img = img.astype(np.float32, copy=False)
    img = img / 255.0
    img = img.reshape(1, img.shape[0], img.shape[1], 1)

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(output_dir + '.meta')
        loader.restore(sess, output_dir)

        # Get Tensors from loaded model
        loaded_input_img = loaded_graph.get_tensor_by_name('input_image:0')
        loaded_density_map = loaded_graph.get_tensor_by_name('density_map:0')

        density_map = sess.run(loaded_density_map, feed_dict={loaded_input_img: img})

        cv2.imshow('count: {}'.format(np.sum(density_map)), img_org)


if __name__ == '__main__':
    predict_crowd_count('test.jpg')
    # input('any input to quit')
    # cv2.destroyAllWindows()