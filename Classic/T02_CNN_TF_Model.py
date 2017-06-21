import time
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

dataDir = "D:\\PythonWorkspace\\DataSets\\MNIST_data\\"

# ==================================
#   主要添加了：
#       计时功能；
#       标量和高维矢量和图像的汇总功能（TensorBoard）；
#       学习率递减功能
#
#   模型：LeNet
#
# ==================================

# step 1 -- 读入数据
datasets=1
nIn = 784   # 28*28 = 784
nOut = 10

def ReadDataSet():
    global datasets
    datasets = input_data.read_data_sets(dataDir, one_hot=True)

def Create_Weight_Variable(shape, name=None):
    init = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(init)

def Create_Bias_Variable(shape, name=None):
    init = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(init)

def Conv2D(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def Max_Pool_2X2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def CreateConvolutionLayer(nameScope, inputLayer, width, height, channelsIn, channelsOut, isPooled,
                           shouldImage=False, shouldWeight=False, shouldBias=False):
    with tf.name_scope(nameScope) as scope:
        weights = Create_Weight_Variable([width, height, channelsIn, channelsOut], name="weight")
        if shouldWeight:
            tf.summary.histogram("weights", weights)
        biases = Create_Bias_Variable([channelsOut], name="bias")
        if shouldBias:
            tf.summary.histogram("biases", biases)
        hidden = tf.nn.relu(Conv2D(inputLayer, weights) + biases, name="hidden-Layer")
        if shouldImage:
            weights_r = tf.transpose(weights, perm=[3,0,1,2], name="kernal")
            tf.summary.image("convKernals", weights_r, max_outputs=32)
        if isPooled:
            return Max_Pool_2X2(hidden)
        else:
            return hidden

def CreateNormalLayer(nameScope, inputLayer, nodesIn, nodesOut,
                      shouldImageWeight=False):
    with tf.name_scope(nameScope) as scope:
        weights = Create_Weight_Variable([nodesIn, nodesOut])
        biases = Create_Bias_Variable([nodesOut])
        result = tf.nn.relu(tf.matmul(inputLayer, weights) + biases)
        if shouldImageWeight:
            return result, weights
        else:
            return result

def model(isTraining=True):
    # step 2 -- 设置占位符与变量
    X = tf.placeholder(tf.float32, [None, nIn])  # 每个图像为784的向量表示
    Y = tf.placeholder(tf.float32, [None, nOut])  # one-hot表示的类别

    # step 3 -- 设计模型，设计Graph
    x_image = tf.reshape(X, [-1, 28, 28, 1])
    # 第一层卷积
    hiddenLayer1 = CreateConvolutionLayer(
        "hidden1", x_image, 5, 5, 1, 32, True,
        shouldWeight=False, shouldBias=False, shouldImage=True)
    # 第二层卷积
    hiddenLayer2 = CreateConvolutionLayer(
        "hidden2", hiddenLayer1, 5, 5, 32, 64, True)
    hiddenLayer2_flat = tf.reshape(hiddenLayer2, [-1, 7 * 7 * 64])
    # 密集连接层
    hiddenLayer3 = CreateNormalLayer(
        "hidden3_normal", hiddenLayer2_flat, 7 * 7 * 64, 1024)
    # Dropout
    if isTraining:
        keep_prob = tf.placeholder(tf.float32)
        hiddenLayer3_drop = tf.nn.dropout(hiddenLayer3, keep_prob)
        # 输出层
        y_conv = CreateNormalLayer(
            "output", hiddenLayer3_drop, 1024, 10)
        return X, Y, y_conv, keep_prob
    else:
        # 输出层
        y_conv, weights_image = CreateNormalLayer(
            "output", hiddenLayer3, 1024, 10, shouldImageWeight=True)
        return X, Y, y_conv, weights_image



