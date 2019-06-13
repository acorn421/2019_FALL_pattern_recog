'''
2019-05-14
credit by Seungho Jeon

the basic usage of RNN with various cell types, such as Basic RNN cell, LSTM cell, or GRU cell
also, I add several codes for using tensorboard.
so you can visualize the learning progress of your model.

although I originally wrote this source codes using python3.6 and tensorflow-1.13.0,
I strongly recommend using tensorboard-1.12.*.
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets

class RNN:
    def __init__(self, n_in, n_out, n_timestep, n_hidden, cell_type='basic'):
        self.n_in = n_in
        self.n_out = n_out
        self.n_timestep = n_timestep
        self.n_hidden = n_hidden
        self.cell_type = cell_type

    def build(self):
        self.x = tf.placeholder(tf.float32, [None, self.n_timestep, self.n_in])
        self.t = tf.placeholder(tf.float32, [None, self.n_out])
        self.lr = tf.placeholder(tf.float32, [])

        if self.cell_type == 'basic':
            cell = rnn.BasicRNNCell(self.n_hidden)
        elif self.cell_type == 'lstm':
            cell = rnn.LSTMCell(self.n_hidden)
        elif self.cell_type == 'gru':
            cell = rnn.GRUCell(self.n_hidden)

        inputs = tf.unstack(self.x, self.n_timestep, 1)
        W = tf.get_variable('W_output', dtype=tf.float32, initializer=tf.random_normal([self.n_hidden, self.n_out], stddev=0.1))
        b = tf.get_variable('b_output', dtype=tf.float32, initializer=tf.zeros([self.n_out]))

        outputs, states = tf.nn.static_rnn(cell, inputs, dtype=tf.float32)
        y = tf.matmul(outputs[-1], W) + b

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.t, logits=y))
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

        correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(self.t, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        tf.summary.histogram('y', y)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()


def main():
    mnist = read_data_sets('./MNIST/', one_hot=True)
    n_in = 28
    n_out = 10
    n_hidden = 128
    n_timestep = 28
    n_batch = 100
    cell_type = 'gru'
    lr = 1e-1

    with tf.Session() as sess:
        model = RNN(n_in, n_out, n_timestep, n_hidden, cell_type)
        model.build()
        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter('./board/test1', sess.graph)

        updates = list()
        tr_accuracies = list()
        te_accuracies = list()
        fig, ax = plt.subplots(1)
        for i in range(1000):
            tr_x, tr_t = mnist.train.next_batch(n_batch)
            tr_x = tr_x.reshape((n_batch, 28, 28))

            summary, loss, tr_acc, _ = sess.run([model.merged, model.loss, model.accuracy, model.optimizer], feed_dict={model.x:tr_x, model.t:tr_t, model.lr:lr})
            writer.add_summary(summary, i)

            if i % 10 == 0:
                te_acc = sess.run(model.accuracy, feed_dict={model.x: mnist.test.images.reshape((-1, 28, 28)), model.t: mnist.test.labels})
                updates.append(i)
                tr_accuracies.append(tr_acc)
                te_accuracies.append(te_acc)

                print(i, ', loss: ', loss, ', train accuracy: ', tr_acc, ', test accuracy: ', te_acc)
                ax.cla()
                ax.plot(updates, tr_accuracies, c='b', label='train accuracy')
                ax.plot(updates, te_accuracies, c='r', label='test accuracy')
                ax.legend()
                fig.canvas.draw()
                plt.pause(0.1)
        plt.show()

if __name__ == '__main__':
    main()