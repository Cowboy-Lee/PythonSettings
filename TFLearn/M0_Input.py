
import tensorflow.examples.tutorials.mnist.input_data as input_data

dataDir = "D:\\PythonWorkspace\\DataSets\\MNIST_data\\"


def ReadDataSet():
    dataSet = input_data.read_data_sets(dataDir, one_hot=True)
    return dataSet.train.images, dataSet.train.labels, dataSet.test.images, dataSet.test.labels

