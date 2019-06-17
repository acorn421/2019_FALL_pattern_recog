# Dataset URL : http://cs231n.stanford.edu/tiny-imagenet-200.zip



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