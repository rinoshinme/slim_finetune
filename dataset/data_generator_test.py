import cv2
import numpy as np
import tensorflow as tf
from dataset.data_generator import ImageDataGenerator


if __name__ == '__main__':
    txt_file = r'F:\\DATASET2019\baokong09_20190717\train.txt'
    data_iterator = ImageDataGenerator(txt_file, 'training', 1, 9, shuffle=True)

    data_next_batch = data_iterator.iterator.get_next()
    with tf.Session() as sess:
        for i in range(20):
            x, y = sess.run(data_next_batch)

            x = x[0, :, :, :]
            x = x + np.array([121.55213, 113.84197, 99.5037])
            x = np.array(x, dtype=np.uint8)

            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            cv2.imwrite('./image{}.jpg'.format(i), x)
