import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets

# fully connected hidden layer
def fully_conn(x, n_hid, name):
    n_batch, n_dim = x.shape
    W = tf.get_variable(name + '_W', dtype=tf.float32, initializer=tf.random_normal([int(n_dim), n_hid], stddev=0.1))
    b = tf.get_variable(name + '_b', dtype=tf.float32, initializer=tf.zeros([n_hid]))

    return tf.matmul(x, W) + b

# ordinary Variational Autoencoder
class M1:
    def __init__(self, n_in, n_z):
        self.n_in = n_in
        self.n_z = n_z

    def _build_encoder(self, x):
        with tf.variable_scope('M1_encoder', reuse=tf.AUTO_REUSE):
            h1 = tf.nn.softplus(fully_conn(x, int(self.n_in / 2), 'M1_encoder1'))
            mu = fully_conn(h1, self.n_z, 'M1_mu')
            sig = tf.nn.softplus(fully_conn(h1, self.n_z, 'M1_sig'))

        return mu, sig

    def _build_decoder(self, z):
        with tf.variable_scope('M1_decoder', reuse=tf.AUTO_REUSE):
            h1 = tf.nn.softplus(fully_conn(z, int(self.n_in / 2), 'M1_decoder1'))
            y = fully_conn(h1, self.n_in, 'M1_output')

        return y

    def ELBO(self, mu, sig, y):
        loss_recon = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x, logits=y), 1))
        loss_kld = tf.reduce_mean(tf.reduce_sum(tf.distributions.kl_divergence(self._get_dist(mu, sig),
                                                                               self._get_dist(tf.zeros([self.n_z]),
                                                                                              tf.ones([self.n_z]))), 1))
        return loss_recon + 0.5 * loss_kld

    def build(self):
        self.x = tf.placeholder(tf.float32, [None, self.n_in])
        self.lr = tf.placeholder(tf.float32, [])

        mu, sig = self._build_encoder(self.x)
        e = tf.random_normal(tf.shape(mu))
        z = mu + tf.multiply(e, sig)

        y = self._build_decoder(z)

        self.loss = self.ELBO(mu, sig, y)
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _get_dist(self, mu, sig):
        dist = tf.distributions.Normal(loc=mu, scale=sig)
        return dist

class M2:
    def __init__(self, n_in, n_z, n_cls):
        self.n_in = n_in
        self.n_z = n_z
        self.n_cls = n_cls

    def _build_classifier(self, x):
        with tf.variable_scope('M2_classifier', reuse=tf.AUTO_REUSE):
            h1 = tf.nn.softplus(fully_conn(x, int(self.n_in / 2), 'M2_classifier1'))
            pred = fully_conn(h1, self.n_cls, 'M2_classifier2')
        return pred

    def _build_encoder(self, x):
        with tf.variable_scope('M2_encoder', reuse=tf.AUTO_REUSE):
            h1 = tf.nn.softplus(fully_conn(x, int(self.n_in / 2), 'M2_encoder1'))
            mu = fully_conn(h1, self.n_z, 'M2_mu')
            sig = tf.nn.softplus(fully_conn(h1, self.n_z, 'M2_sig'))
        return mu, sig

    def _build_decoder(self, z):
        with tf.variable_scope('M2_decoder', reuse=tf.AUTO_REUSE):
            h1 = tf.nn.softplus(fully_conn(z, int(self.n_in / 2), 'M2_decode1'))
            y = fully_conn(h1, self.n_in, 'M2_output')
        return y

    def _pathway(self, x_l, label):
        x_with_label = tf.concat([x_l, label], 1)
        mu, sig = self._build_encoder(x_with_label)

        e = tf.random_normal(tf.shape(mu))
        z = mu = tf.multiply(e, sig)
        y = self._build_decoder(z)

        return mu, sig, z, y

    def total_loss(self, mu_l, sig_l, z_l, y_l, mu_u, sig_u, z_u, y_u):
        # TODO: write the codes that calcuate the loss
        # loss = Loss_labelled + Loss_unlabelled + alpha * cross_entropy

        loss_recon = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x, logits=y), 1))
        loss_kld = tf.reduce_mean(tf.reduce_sum(tf.distributions.kl_divergence(self._get_dist(mu, sig),
                                                                               self._get_dist(tf.zeros([self.n_z]),
                                                                                              tf.ones([self.n_z]))), 1))

        return None

    def build(self):
        self.x_l = tf.placeholder(tf.float32, [None, self.n_in])        # labelled data
        self.x_u = tf.placeholder(tf.float32, [None, self.n_in])        # unlabelled data
        self.label = tf.placeholder(tf.float32, [None, self.n_cls])     # label corresponding to labelled data
        self.lr = tf.placeholder(tf.float32, [])                         # learning rata

        self.pred_l = self._build_classifier(self.x_l)
        self.pred_u = self._build_classifier(self.x_u)
        mu_l, sig_l, z_l, y_l = self._pathway(self.x_l, self.label)     # forward path for labelled data
        mu_u, sig_u, z_u, y_u = self._pathway(self.x_u, self.pred_u)      # forward path for unlabelled data

        self.loss = self.total_loss(mu_l, sig_l, z_l, y_l, mu_u, sig_u, z_u, y_u)                                   # caclcuating loss(need to implement)
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)  # optimizing the model

def main():
    # hyper parameters
    # it is fine to change the value if you want to
    n_in = 784
    n_z1 = 128
    n_z2 = 32
    n_cls = 10
    lr = 1e-2
    mnist = read_data_sets('./MNIST-data/', one_hot=True)

    with tf.Session() as sess:
        m1 = M1(n_in, n_z1)
        m1.build()
        m2 = M2(n_z1, n_z2, n_cls)
        m2.build()
        sess.run(tf.global_variables_initializer())

        # TODO: train M1
        # need to implement

        # TODO: train M2
        # z1 = m1.z
        # m2.x = z1
        # need to implement


if __name__  == '__main__':
    main()
