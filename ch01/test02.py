import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, activation_function=None):
    """
    神经层函数
    :param inputs:输入值
    :param in_size:输入的大小
    :param out_size:输出的大小
    :param activation_function:激励函数
    :return:
    """
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases  # Wx_plus_b是神经网络未激活的值，tf.matmul()是矩阵的乘法

    """
    当激励函数为空时，Wx_plus_b输出就是当前的预测值，
    不为None时候，就把Wx_plus_b传入到激励函数中去
    """
    if activation_function is None:
        output = Wx_plus_b
    else:
        output = activation_function(Wx_plus_b)

    return output


if __name__ == '__main__':
    # 1.准备数据
    x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
    y_data = np.square(x_data) - 0.5 + noise

    # 2.利用占位符定义我们所需的神经网络的输入
    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])

    # 3.定义神经层（输入层、隐藏层、输出层）
    # 构建输入层1个，隐藏层10个，输出层1个的神经网络

    # 3.1定义隐藏层
    l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
    # 3.定义输出层（此时的输入就是隐藏层的输出——l1）
    prediction = add_layer(l1, 10, 1, activation_function=None)

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))

    # 让机器学习提升它的准确率
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 初始化变量
    init = tf.global_variables_initializer()

    # 定义session
    # with tf.Session() as sess:
    #     sess.run(init)
    sess = tf.Session()
    sess.run(init)
    # 训练
    for i in range(2000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
