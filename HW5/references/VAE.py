import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
from tensorflow.contrib.distributions import MultivariateNormalDiag

class VAE:
    def __init__(self, n_in):
        self.n_in = n_in

    def _build_encoder(self, x):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            W_en1 = tf.get_variable('W_en1', dtype=tf.float32, initializer=tf.random_normal([self.n_in, 128], stddev=0.1))
            W_mu = tf.get_variable('W_mu', dtype=tf.float32, initializer=tf.random_normal([128, 2], stddev=0.1))
            W_sig = tf.get_variable('W_sig', dtype=tf.float32, initializer=tf.random_normal([128, 2], stddev=0.1))
            b_en1 = tf.get_variable('b_en1', dtype=tf.float32, initializer=tf.zeros([128]))
            b_mu = tf.get_variable('b_mu', dtype=tf.float32, initializer=tf.zeros([2]))
            b_sig = tf.get_variable('b_sig', dtype=tf.float32, initializer=tf.zeros([2]))

            h1 = tf.nn.relu(tf.matmul(x, W_en1) + b_en1)
            mu = tf.matmul(h1, W_mu) + b_mu
            sig = tf.matmul(h1, W_sig) + b_sig

        return mu, sig

    def _build_decoder(self, z):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            W_de1 = tf.get_variable('W_de1', dtype=tf.float32, initializer=tf.random_normal([2, 128], stddev=0.1))
            W_de2 = tf.get_variable('W_de2', dtype=tf.float32, initializer=tf.random_normal([128, self.n_in], stddev=0.1))
            b_de1 = tf.get_variable('b_de1', dtype=tf.float32, initializer=tf.zeros([128]))
            b_de2 = tf.get_variable('b_de2', dtype=tf.float32, initializer=tf.zeros([self.n_in]))

            h2 = tf.nn.relu(tf.matmul(z, W_de1) + b_de1)
            y = tf.matmul(h2, W_de2) + b_de2

        return y


    def build(self):
        self.x = tf.placeholder(tf.float32, [None, self.n_in])
        self.lr = tf.placeholder(tf.float32, [])
        self.z_in = tf.placeholder(tf.float32, [None, 2])

        mu, sig = self._build_encoder(self.x)
        e = tf.random_normal(tf.shape(mu))
        z = mu + tf.multiply(e, tf.nn.softplus(sig))
        y = self._build_decoder(z)
        gen = self._build_decoder(self.z_in)

        self.z = z
        self.mu = mu
        self.sig = tf.nn.softplus(sig)
        self.recon = tf.nn.sigmoid(y)
        self.fake = tf.nn.sigmoid(gen)

        loss_recon = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x, logits=y), 1))
        loss_kld = tf.reduce_mean(tf.reduce_sum(tf.distributions.kl_divergence(self._get_dist(mu, tf.nn.softplus(sig)), self._get_dist(tf.zeros([2]), tf.ones([2]))), 1))
        self.kld = loss_kld

        self.loss = loss_recon + 0.5 * loss_kld
        self.train = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _kld(self, q, p, e=1e-24):
        return q * tf.log(q + e) - q * tf.log(p + e)

    def _get_dist(self, mu, sig):
        dist = tf.distributions.Normal(loc=mu, scale=sig)
        return dist


def main():
    mnist = read_data_sets('./MNIST-data/', one_hot=True)
    n_in = 784
    lr = 1e-2

    with tf.Session() as sess:
        model = VAE(n_in)
        model.build()
        sess.run(tf.global_variables_initializer())

        X = np.linspace(-5., 5., 20)
        Y = np.linspace(-5., 5., 20)
        XX, YY = np.meshgrid(X, Y)
        coords = np.dstack((XX, YY)).reshape((-1, 2))
        print(coords.shape)

        xs = list()
        losses = list()
        fig, ax = plt.subplots(1)
        canvas = np.empty((20 * 28, 20 * 28))
        for i in range(10000):
            tr_x, _ = mnist.train.next_batch(100)
            fake, z, kld, mu, sig, recon, loss, _ = sess.run([model.fake, model.z, model.kld, model.mu, model.sig, model.recon, model.loss, model.train], feed_dict={model.x:tr_x, model.lr:lr, model.z_in:coords})

            if i % 100 == 0:
                print(i, loss)
                xs.append(i)
                losses.append(loss)

                ax.cla()

                for h in range(20):
                    for w in range(20):
                        canvas[h * 28 : (h+1) * 28, w * 28 : (w+1) * 28] = fake[h * 20 + w].reshape((28, 28))
                ax.imshow(canvas, cmap='gray')
                fig.canvas.draw()
                plt.pause(0.1)

        plt.show()


if __name__ == '__main__':
    main()