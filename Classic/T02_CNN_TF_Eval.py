# -*- coding=UTF-8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf
import Classic.T02_CNN_TF_Model as mnist
import numpy as np

sess = tf.InteractiveSession()

# ====================================================
#   这里使用 tf的图像处理函数
# ====================================================
image_raw_data = tf.gfile.FastGFile(mnist.dataDir+"\\TestImage.jpg", 'r').read()
image_data = tf.image.decode_jpeg(image_raw_data)
image_data = tf.image.convert_image_dtype(image_data, tf.float32)
originImage = image_data.eval()
image_data = tf.image.resize_images(image_data, [28,28], method=0)

image_data = tf.image.adjust_saturation(image_data, -100)
image_data = tf.image.rgb_to_grayscale(image_data)

image_data = tf.subtract(1.0, image_data)

image_data_to_eval = tf.reshape(image_data, [1, 28*28])
image_data_to_eval = image_data_to_eval.eval()


X, Y, y_conv, wImages = mnist.model(isTraining=False)


# top_k = tf.nn.in_top_k(y_conv, Y, k=3)

saver = tf.train.Saver()

with tf.Session() as sess:
    # Restores from checkpoint
    saver.restore(sess, mnist.dataDir + "\\MNIST_result.ckpt-19999_20170326_2")

    wImages = wImages.eval()

    weights_data = np.reshape(wImages, [32, 32, 1, 10])
    weights_image = weights_data.transpose((3,0,1,2))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.title(u"企图显示数字0的特征权重，但是失败了")
    for i in range(10):
        weights_image_one = weights_image[i,:,:,:]
        plt.subplot(3,4,i+1)
        plt.imshow(weights_image_one.squeeze())
    plt.show()

    answer = tf.argmax(y_conv, axis=1)
    probabilities, finalAnswer = sess.run([y_conv, answer], feed_dict={X:image_data_to_eval})
    print(probabilities)
    print("answer is: ", finalAnswer)
    image_data_to_show = image_data.eval()
    plt.imshow(image_data_to_show.squeeze())
    plt.show()

