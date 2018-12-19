import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



# 读取MNIST数据
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


sess = tf.Session()

# 定义placeholder x,y_
# x深度为1
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# 将x转换为28 * 28 图像
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 定义第一个卷积层
bias1 = tf.Variable(tf.constant(0.1, shape=[32]))
w1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=0.1))
layer1 = tf.nn.conv2d(x_image, w1, strides=[1, 1, 1, 1], padding="SAME")
conv1 = tf.nn.relu(layer1 + bias1)
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
w2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=0.1))
layer2 = tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding="SAME")
conv2 = tf.nn.relu(layer2 + bias2)
pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# 此时pool2 维度为[1, 7, 7, 64]
# 转换为1维
layer2_flat = tf.reshape(pool2, [-1, 7*7*64])


# 增加一个输出为1024个参数的全连接层
w3 = tf.Variable(tf.truncated_normal(shape=[7*7*64, 1024], stddev=0.1))
bias3 = tf.Variable(tf.constant(0.1, shape=[1024]))
layer3 = tf.nn.relu(tf.matmul(layer2_flat, w3) + bias3)
# 使用dropout
keep_prob = tf.placeholder(tf.float32)
layer3_drop = tf.nn.dropout(layer3, keep_prob)

# 输出层将1024转换为10维，对应10个类别
w4 = tf.Variable(tf.truncated_normal(shape=[1024, 10], stddev=0.1))
bias4 = tf.Variable(tf.constant(0.1, shape=[10]))
y = tf.matmul(layer3_drop, w4) + bias4
# 使用dropout


# 定义交叉熵
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# 同样定义train_step
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

# 定义准确性
prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))


# 所有变量初始化
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(5000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 计算对测试集的准确率
test_accuracy = accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
print(" test accuracy %g" % test_accuracy)










