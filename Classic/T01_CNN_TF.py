# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import time

dataDir = "D:\\PythonWorkspace\\DataSets\\MNIST_data\\"


# ==================================
#   主要添加了：
#       计时功能；
#       标量和高维矢量和图像的保存功能（TensorBoard）；
#       学习率递减功能
# ==================================

# step 1 -- 读入数据
mnist = input_data.read_data_sets(dataDir, one_hot=True)
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
nIn = 784   # 28*28 = 784
nOut = 10

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
            tf.summary.image("convKernals", weights_r, max_outputs=10)
        if isPooled:
            return Max_Pool_2X2(hidden)
        else:
            return hidden

def CreateNormalLayer(nameScope, inputLayer, nodesIn, nodesOut,
                      shouldImageWeight=False, imageSize=10):
    with tf.name_scope(nameScope) as scope:
        weights = Create_Weight_Variable([nodesIn, nodesOut])
        if shouldImageWeight:
            weights_data = tf.reshape(weights, [imageSize, imageSize, 1, nodesOut])
            weights_image = tf.transpose(weights_data, perm=[3,0,1,2])
            tf.summary.image("imageJudger", weights_image, max_outputs=nodesOut)
        biases = Create_Bias_Variable([nodesOut])
        return tf.nn.relu(tf.matmul(inputLayer, weights) + biases)

# step 2 -- 设置占位符与变量
X = tf.placeholder("float", [None, nIn])#每个图像为784的向量表示
Y = tf.placeholder("float", [None, nOut])#one-hot表示的类别
global_step = tf.Variable(0, trainable=False)   #全局步骤，在优化时自动添加
# learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96)
# tf.summary.scalar("learning_rate", learning_rate)

# step 3 -- 设计模型，设计Graph
x_image = tf.reshape(X, [-1, 28, 28, 1])
# 第一层卷积
hiddenLayer1 = CreateConvolutionLayer(
    "hidden1", x_image, 5, 5, 1, 32, True,
    shouldWeight=True, shouldBias=True)
# 第二层卷积
hiddenLayer2 = CreateConvolutionLayer(
    "hidden2", hiddenLayer1, 5, 5, 32, 64, True)
hiddenLayer2_flat = tf.reshape(hiddenLayer2, [-1,7*7*64])
# 密集连接层
hiddenLayer3 = CreateNormalLayer(
    "hidden3_normal", hiddenLayer2_flat, 7*7*64, 1024)
# Dropout
keep_prob = tf.placeholder("float")
hiddenLayer3_drop = tf.nn.dropout(hiddenLayer3, keep_prob)
#输出层
y_conv =  CreateNormalLayer(
    "output", hiddenLayer3_drop, 1024, 10, shouldImageWeight=False, imageSize=32)

# step 4 -- 定义优化目标、模型评估方式
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_conv), name="xEntropy") # 计算预测与真实的交叉熵
train_op = tf.train.AdamOptimizer(1e-4).minimize(cost, global_step=global_step)
correct_prediction = tf.equal(tf.argmax(y_conv,axis=1), tf.argmax(Y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")
tf.summary.scalar(accuracy.op.name, accuracy)
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(dataDir, graph_def=sess.graph_def)
saver = tf.train.Saver()

lastTime = time.time()
# step 5 -- 开始优化
sess.run(tf.global_variables_initializer())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%10==0:
        train_accuracy, s_op = sess.run([accuracy,summary_op], feed_dict={X:batch[0], Y:batch[1], keep_prob:1.0})
        summary_writer.add_summary(s_op, i)
        now = time.time()
        print("i=%d, train_accuracy= %g,  elapsed time= %.1f sec" % (i,train_accuracy, now-lastTime))
        lastTime=now
    sess.run(train_op, feed_dict={X:batch[0], Y:batch[1], keep_prob:0.5})
    if (i%10000==0):
        saver.save(sess, dataDir, global_step=i)

def GetTestAccurate():
    test_batch1 = mnist.test.next_batch(2500)
    return sess.run(accuracy, feed_dict={X: test_batch1[0], Y: test_batch1[1], keep_prob: 1.0})
print("calculating test accuracy...")
test_acc = (GetTestAccurate()+GetTestAccurate()+GetTestAccurate()+GetTestAccurate())/4.0
# test_acc = sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels, keep_prob:1.0})
print("test accuracy: %g" % test_acc)

sess.close()

