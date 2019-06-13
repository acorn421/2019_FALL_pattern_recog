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
        # modify original code
        # z = mu + tf.multiply(e, sig)
        self.z = mu + tf.multiply(e, sig)

        # modify original code
        # y = self._build_decoder(self.z)
        self.y = self._build_decoder(self.z)

        self.loss = self.ELBO(mu, sig, self.y)
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

    def _loss_labelled(self, x, y, mu, sig, label):
        loss_recon = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=y), 1))
        loss_kld = tf.reduce_mean(tf.reduce_sum(tf.distributions.kl_divergence(self._get_dist(mu, sig), self._get_dist(tf.zeros([self.n_z]), tf.ones([self.n_z]))), 1))
        prior_y = (1. / self.n_cls) * tf.ones_like(label)
        loss_logpy = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels = prior_y, logits = label), 1))
        loss_labbeld = loss_recon + loss_kld + loss_logpy

        return loss_labbeld


    def total_loss(self, mu_l, sig_l, z_l, y_l, mu_u, sig_u, z_u, y_u):
        # TODO: write the codes that calcuate the loss
        # loss = Loss_labelled + Loss_unlabelled + alpha * cross_entropy

        # Loss_labelled
        loss_l = tf.reduce_mean(self._loss_labelled(self.x_l, y_l, mu_l, sig_l, self.label))
        # Loss_unlabelled
        loss_u = None
        for i in range(self.n_cls):
            _loss_u = self._loss_labelled(self.x_u, y_u[i], mu_u[i], sig_u[i], self.label_u[i])
            if i==0:
                _loss_u = tf.reshape(_loss_u, [1, 1])
                # loss_u = tf.expand_dims(_loss_u, 1)
                loss_u = _loss_u
            else:
                # _loss_u = tf.expand_dims(_loss_u, 1)
                _loss_u = tf.reshape(_loss_u, [1, 1])
                # loss_u = tf.concat([loss_u, tf.reshape(_loss_u, [1,1])], 1)  
                loss_u = tf.concat([loss_u, _loss_u], 1)
        
        self.prob_u = tf.nn.softmax(self.pred_u, dim=-1)
        loss_u = tf.multiply(self.prob_u, tf.subtract(loss_u, -tf.log(self.prob_u)))
        loss_u = tf.reduce_mean(tf.reduce_sum(loss_u, 1))
        # Loss classification
        loss_clf = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.label, logits = self.pred_l))
        # alpha
        alpha = 0.1*100


        self.loss_l = loss_l
        self.loss_u = loss_u
        self.loss_clf = loss_clf

        loss = loss_l + loss_u + alpha * loss_clf

        return loss

    def build(self):
        self.x_l = tf.placeholder(tf.float32, [None, self.n_in])        # labelled data
        self.x_u = tf.placeholder(tf.float32, [None, self.n_in])        # unlabelled data
        self.label = tf.placeholder(tf.float32, [None, self.n_cls])     # label corresponding to labelled data
        self.lr = tf.placeholder(tf.float32, [])                         # learning rata

        self.pred_l = self._build_classifier(self.x_l)
        self.pred_u = self._build_classifier(self.x_u)
        mu_l, sig_l, z_l, y_l = self._pathway(self.x_l, self.label)     # forward path for labelled data

        # Modify original source code
        # mu_u, sig_u, z_u, y_u = self._pathway(self.x_u, self.pred_u)      # forward path for unlabelled data
        mu_u = [0] * self.n_cls
        sig_u = [0] * self.n_cls
        z_u = [0] * self.n_cls
        y_u = [0] * self.n_cls
        # self.label_u = tf.one_hot(list(range(self.n_cls)), self.n_cls)
        self.label_u = [0] * self.n_cls
        # print(self.label_u[0])
        for i in range(self.n_cls):
            _y = i * tf.ones([tf.shape(self.x_u)[0]], tf.int32)
            self.label_u[i] = tf.one_hot(_y, self.n_cls)
            mu_u[i], sig_u[i], z_u[i], y_u[i] = self._pathway(self.x_u, self.label_u[i])      # forward path for unlabelled data
            
        
        self.loss = self.total_loss(mu_l, sig_l, z_l, y_l, mu_u, sig_u, z_u, y_u)           # caclcuating loss(need to implement)
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)  # optimizing the model

    def _get_dist(self, mu, sig):
        dist = tf.distributions.Normal(loc=mu, scale=sig)
        return dist

def main():
    # hyper parameters
    # it is fine to change the value if you want to
    n_in = 784
    n_z1 = 128
    n_z2 = 32
    n_cls = 10
    # modify original code
    # lr = 1e-2
    lr_1 = 1e-2
    lr_2 = 1e-5
    mnist = read_data_sets('./MNIST-data/', one_hot=True)

    with tf.Session() as sess:
        m1 = M1(n_in, n_z1)
        m1.build()
        m2 = M2(n_z1, n_z2, n_cls)
        m2.build()
        sess.run(tf.global_variables_initializer())

        # TODO: train M1
        # need to implement
        fig, ax = plt.subplots(1, 3)
        xs1 = list()
        losses1 = list()
        # for i in range(10000):
        for i in range(10000):
            tr_x, _ = mnist.train.next_batch(100)
            loss, y, _ = sess.run([m1.loss, m1.y, m1.opt], feed_dict={m1.x:tr_x, m1.lr:lr_1})

            if i % 100 == 0:
                print(i, loss)
                xs1.append(i)
                losses1.append(loss)

                ax[0].cla()
                ax[0].imshow(y[0].reshape(28, 28), cmap='gray')
                ax[1].set_title('recon img')
                ax[1].cla()
                ax[1].plot(xs1, losses1, c='r')
                ax[1].set_title('m1 loss')

                fig.canvas.draw()
                plt.pause(0.1)

        # TODO: train M2
        # z1 = m1.z
        # m2.x = z1
        # need to implement
        xs2 = list()
        losses2 = list()
        for i in range(10000):
            tr_l_x, tr_l_y = mnist.train.next_batch(50)
            tr_u_x, _ = mnist.train.next_batch(50)

            z1_l = sess.run(m1.z, feed_dict={m1.x:tr_l_x, m1.lr:lr_1})
            z1_u = sess.run(m1.z, feed_dict={m1.x:tr_u_x, m1.lr:lr_1})

            

            loss, loss_l, loss_u, loss_clf, _ = sess.run([m2.loss, m2.loss_l, m2.loss_u, m2.loss_clf, m2.opt], feed_dict={m2.x_l:z1_l, m2.x_u:z1_u, m2.label:tr_l_y, m2.lr:lr_2})

            if i % 100 == 0:
                print(i, loss, loss_l, loss_u, loss_clf)
                xs2.append(i)
                losses2.append(loss)

                ax[2].cla()
                ax[2].plot(xs2, losses2, c='r')
                ax[2].set_title('m2 loss')

                fig.canvas.draw()
                plt.pause(0.1)
        plt.show()

if __name__  == '__main__':
    main()
