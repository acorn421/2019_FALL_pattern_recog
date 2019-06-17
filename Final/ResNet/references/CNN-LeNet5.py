import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets

class LeNet5:
    def __init__(self, n_h, n_w, n_c, n_out):
        self.n_h = n_h
        self.n_w = n_w
        self.n_c = n_c
        self.n_out = n_out

    def build(self):
        self.x = tf.placeholder(tf.float32, [None, self.n_h, self.n_w, self.n_c])
        self.t = tf.placeholder(tf.float32, [None, self.n_out])
        self.lr = tf.placeholder(tf.float32, [])
        n_batch = tf.shape(self.x)[0]

        h1 = self.conv2d(self.x, [5, 5], 6, tf.nn.tanh, 'layer_1')
        h2 = self.max_pool(h1, 'layer_2')
        h3 = self.conv2d(h2, [5, 5], 16, tf.nn.tanh, 'layer_3')
        h4 = self.max_pool(h3, 'layer_4')

        (_, h, w, c) = h4.shape
        fc0 = tf.reshape(h4, [n_batch, h * w * c])

        fc1 = self.fully_conn(fc0, 120, tf.nn.tanh, 'layer_5')
        fc2 = self.fully_conn(fc1, 84, tf.nn.tanh, 'layer_6')
        y = self.fully_conn(fc2, self.n_out, tf.identity, 'layer_output')

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.t, logits=y))
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

        correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(self.t, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def resnet_layer_conv2d(self, x, kernel_shape, n_output, activation, last=False, name=None):
        h = self.conv2d(x, kernel_shape, n_output, activation, name + '_1')
        h = self.conv2d(h, kernel_shape, n_output, tf.identity, name + '_2')

        shortcut = self.conv2d(x, [1, 1], n_output, tf.identity, name + '_shortcut')

        if last:
            h = h + shortcut
        else:
            h = activation(h + shortcut)
        return h

    def max_pool(self, x, name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name=name)

    def conv2d(self, x, kernel_shape, n_output, activation, name):
        W = tf.get_variable(name + '_W', dtype=tf.float32,
                            initializer=tf.random_normal([kernel_shape[0], kernel_shape[1], int(x.shape[3]), n_output],
                                                         stddev=0.1))
        b = tf.get_variable(name + '_b', dtype=tf.float32, initializer=tf.zeros([n_output]))
        h = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID') + b
        return activation(h)

    def fully_conn(self, x, n_output, activation, name):
        W = tf.get_variable(name + '_W', dtype=tf.float32,
                            initializer=tf.random_normal([int(x.shape[1]), n_output], stddev=0.1))
        b = tf.get_variable(name + '_b', dtype=tf.float32,
                            initializer=tf.zeros([n_output]))
        h = tf.matmul(x, W) + b
        return activation(h)



def main():
    mnist = read_data_sets('./MNIST', one_hot=True)
    n_h = 28
    n_w = 28
    n_c = 1
    n_out = 10
    n_batch = 100
    lr = 1e-1


    with tf.Session() as sess:
        model = LeNet5(n_h, n_w, n_c, n_out)
        model.build()
        sess.run(tf.global_variables_initializer())

        updates = list()
        tr_accuracies = list()
        te_accuracies = list()
        losses = list()
        fig, ax = plt.subplots(1, 2)
        for i in range(1000):
            tr_x, tr_t = mnist.train.next_batch(n_batch)
            tr_x = tr_x.reshape((-1, n_h, n_w, n_c))

            loss, tr_acc, _ = sess.run([model.loss, model.accuracy, model.optimizer], feed_dict={model.x:tr_x, model.t:tr_t, model.lr:lr})
            if i % 10 == 0:
                print('update: ', i, ', training accuracy: ', tr_acc)
                te_acc = sess.run(model.accuracy, feed_dict={model.x:mnist.test.images.reshape((-1, n_h, n_w, n_c)), model.t:mnist.test.labels})

                updates.append(i)
                tr_accuracies.append(tr_acc)
                te_accuracies.append(te_acc)
                losses.append(loss)

                for j in range(len(ax)):ax[j].cla()
                ax[0].plot(updates, tr_accuracies, c='b', label='training accuracy')
                ax[0].plot(updates, te_accuracies, c='r', label='test accuracy')
                ax[1].plot(updates, losses, c='r')
                ax[0].legend()
                ax[0].set_title('accuracy')
                ax[1].set_title('loss')
                fig.canvas.draw()
                plt.pause(0.1)
        plt.show()



if __name__ == '__main__':
    main()