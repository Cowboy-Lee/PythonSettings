import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

summaryFileDir = "D:\\PythonWorkspace\\DataSets\\MNIST_data"

# ==================================
#
#   模型：简化的AlexNet，只有5层，而且某些激活函数也由relu变成了tanh
#
# ==================================


def model():
    network = input_data(shape=[None, 28, 28, 1], name='input')
    # --------------------------------------第1层
    # CNN中的卷积操作,下面会有详细解释
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    # 最大池化操作
    network = max_pool_2d(network, 2)
    # 局部响应归一化操作 —— 虽然不知道是什么，据说配合ReLU可以有2%的效果提升
    network = local_response_normalization(network)
    #-----------------------------------------第2层
    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    # ------------------------------------------第3层，相当于原版AlexNet的第6层
    # 全连接操作
    network = fully_connected(network, 128, activation='tanh')
    # dropout操作
    network = dropout(network, 0.8)
    # --------------------------------------------第4层，相当于原版AlexNet的第7层
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.8)
    # ----------------------------------------------第5层，输出，相当于原版AlexNet的第8层
    network = fully_connected(network, 10, activation='softmax')
    # 回归操作
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')

    # DNN操作，构建深度神经网络
    model = tflearn.DNN(network,
                    tensorboard_verbose=0,
                    tensorboard_dir=summaryFileDir)
    return model


