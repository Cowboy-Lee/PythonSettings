
import tensorflow as tf
import time

from Classic import T02_CNN_TF_Model as mnist

sess = tf.InteractiveSession()

mnist.ReadDataSet()
X, Y, y_conv, keep_prob = mnist.model()

# global_step = tf.Variable(0, trainable=False)  # 全局步骤，在优化时自动添加
# learning_rate = tf.train.exponential_decay(0.0002, global_step, 10000, 0.5)

# ------------ 定义优化目标、模型评估方式
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_conv), name="xEntropy") # 计算预测与真实的交叉熵
train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)#, global_step=global_step)
correct_prediction = tf.equal(tf.argmax(y_conv,axis=1), tf.argmax(Y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")
tf.summary.scalar(accuracy.op.name, accuracy)


summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(mnist.dataDir, graph_def=sess.graph_def)
saver = tf.train.Saver()
lastTime = time.time()

# ------------- 开始优化
sess.run(tf.global_variables_initializer())
# saver.restore(sess, mnist.dataDir + "\\MNIST_result.ckpt-19999_20170326_2")
for i in range(20000):
    batch = mnist.datasets.train.next_batch(50)
    if i%10==0:
        train_accuracy, s_op = sess.run([accuracy,summary_op], feed_dict={X:batch[0], Y:batch[1], keep_prob:1.0})
        summary_writer.add_summary(s_op, i)
        now = time.time()
        print("i=%d, train_accuracy= %g,  elapsed time= %.1f sec" % (i,train_accuracy, now-lastTime))
        lastTime=now

    sess.run(train_op, feed_dict={X:batch[0], Y:batch[1], keep_prob:0.5})

    if (i==0 or (i+1)%1000==0):
        saver.save(sess, mnist.dataDir+"\\MNIST_result.ckpt", global_step=i)



def GetTestAccurate():
    test_batch1 = mnist.datasets.test.next_batch(2500)
    return sess.run(accuracy, feed_dict={X: test_batch1[0], Y: test_batch1[1], keep_prob: 1.0})
print("calculating test accuracy...")
test_acc = (GetTestAccurate()+GetTestAccurate()+GetTestAccurate()+GetTestAccurate())/4.0
# test_acc = sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels, keep_prob:1.0})
print("test accuracy: %g" % test_acc)

sess.close()


