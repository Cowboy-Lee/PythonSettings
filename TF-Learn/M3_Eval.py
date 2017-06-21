
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
import numpy as np

from TFLearn import M1_Model as m1
import PIL as pil
import PIL.Image as img
import matplotlib.pyplot as plt

# ====================================================================
#   这里不用 tf的图像处理函数，是因为在TFLearn模型外不需要session
# ====================================================================
image_init = img.open("D:\\PythonWorkspace\\DataSets\\MNIST_data\\TestImage.jpg")
image = image_init.resize((28,28))
image = image.convert("L")
imageArr = np.asarray(image)
print("shape: ", imageArr.shape)
imageArr = 1-(imageArr/255.0)
imageArr = np.reshape(imageArr, [1, 28, 28, 1])

# image_raw_data = tf.gfile.FastGFile("D:\\PythonWorkspace\\DataSets\\MNIST_data\\TestImage.jpg", 'r').read()
# image_data = tf.image.decode_jpeg(image_raw_data)
# image_data = tf.image.convert_image_dtype(image_data, tf.float32)
# originImage = image_data.eval()
# image_data = tf.image.resize_images(image_data, [28,28], method=0)
# image_data = tf.image.adjust_saturation(image_data, -100)
# image_data = tf.image.rgb_to_grayscale(image_data)
# image_data = tf.subtract(1.0, image_data)
# image_data_to_eval = tf.reshape(image_data, [1, 28*28]).eval()

model = m1.model()
model.load("D:\\PythonWorkspace\\DataSets\\MNIST_data\\resultCheckpoint_3-43000")
result = model.predict(imageArr)

print("answer is: ", np.argmax(result))
plt.imshow(image_init)
plt.show()

