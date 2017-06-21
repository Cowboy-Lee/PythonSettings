from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

from TFLearn import M0_Input as m0
from TFLearn import M1_Model as m1

# 加载大名顶顶的mnist数据集（http://yann.lecun.com/exdb/mnist/）
from TFLearn import M0_Input as mnist

X, Y, testX, testY = m0.ReadDataSet()
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

model = m1.model()

model.fit({'input': X}, {'target': Y}, n_epoch=50,
          validation_set=({'input': testX}, {'target': testY}),
          snapshot_step=1000, show_metric=True, run_id='convnet_mnist')
model.save("D:\\PythonWorkspace\\DataSets\\MNIST_data\\train_result.model")

