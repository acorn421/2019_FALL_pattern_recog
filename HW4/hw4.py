import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.datasets import make_classification
from sklearn.decomposition import KernelPCA, TruncatedSVD
from sklearn.metrics.pairwise import rbf_kernel
import scipy

parser = argparse.ArgumentParser(prog='hw4', description='Homework4 of Pattern Recognition Spring 2019')
parser.add_argument('algo', type=int, help="Feature Generation algorithm{'Kernel PCA' : 0, 'SVD' : 1, 'Kernel FDA' : 2)", default=0)
parser.add_argument('-n', '--nofigure', action='store_true', help='do not show figure')
parser.add_argument('-s', '--save', action='store_true', help='save figure to png file')
parser.add_argument('-l', '--load', action='store_true', help='load exist dataset')
args = parser.parse_args()

N = 2   # number of classes
M = 200 # number of data for each class
mean = [[2, 2], [5, 5], [8, 8]]    #   means of classification data
cov = [[[2, 0], [0, 2]], [[4, -3], [-3, 4]], [[2, 0], [0, 2]]]   # covariance of classification data

default_opt = [0]
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

class ClassData:
    def __init__(self, data=np.empty(shape=(0, 2)), labels=np.empty(shape=(0, 1)), pair=np.empty(shape=(0, 3)), gridx=None, gridy=None):
        self.data = data
        self.labels = labels
        self.pair = pair

feature_algorithm = ["Kernel PCA", "SVD", "Kernel FDA"]

origin = ClassData()

fig = plt.figure()
fig.set_size_inches((25, 15), forward=False)

def generate_data():
    global origin, train, test

    if args.algo is not 2:
        if args.load:
            origin.data = np.load('datasets/data.npy')
            origin.labels = np.load('datasets/labels.npy')
        else:
            origin.data = np.append(origin.data, np.random.multivariate_normal(mean[0], cov[0], int(M/2)), axis=0)
            origin.data = np.append(origin.data, np.random.multivariate_normal(mean[2], cov[2], int(M/2)), axis=0)
            origin.labels = np.append(origin.labels, np.array([0] * M))
            origin.data = np.append(origin.data, np.random.multivariate_normal(mean[1], cov[1], M), axis=0)
            origin.labels = np.append(origin.labels, np.array([1] * M))

        # np.save('datasets/data', origin.data)
        # np.save('datasets/labels', origin.labels)

    else:
        if args.load:
            origin.data = np.load('datasets/data_fda.npy')
            origin.labels = np.load('datasets/labels_fda.npy')
        else:
            origin.data, origin.labels = make_classification(n_samples=M, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=3, n_clusters_per_class=1)

        # np.save('datasets/data_fda', origin.data)
        # np.save('datasets/labels_fda', origin.labels)

def kernelFDA(data, labels, n_class, gamma=10):
    X = []
    n = []
    K = []
    M = []
    N = []

    sq_dist = scipy.spatial.distance.pdist(data,'sqeuclidean')
    reshape = scipy.spatial.distance.squareform(sq_dist)

    KK = scipy.exp(-reshape*(gamma))

    for i in range(n_class):
        X.append(data[labels==i,:])
        n.append(X[i].shape[0])
        K.append(KK[:,labels==i])
        M.append(np.sum(K[i],axis=1)/float(n[i]))
        I=np.eye(n[i])
        O=1/float(n[i])
        T=I-O
        N.append(np.dot(K[i],np.dot(T,K[i].T)))

    nn = sum(n)
    M_star = np.sum(KK,axis=1)/float(nn)
    MM = (np.dot((M[0]-M_star),(M[0]-M_star).T)/float(n[0])) + (np.dot((M[1]-M_star),(M[1]-M_star).T)/float(n[1])) + (np.dot((M[2]-M_star),(M[2]-M_star).T)/float(n[2]))
    NN = N[0] + N[1] + N[2]
    NNi=np.linalg.inv(NN)
    NNiMM = np.dot(NNi, MM)
    w, v = np.linalg.eig(NNiMM)
    AA = v.T[:n_class-1]
    Y = np.dot(AA, KK)

    return Y.T.astype(np.float_)

def plot_class(data, size, pos, title=None):
    global fig

    color = ('ro', 'go', 'bo')  #   colors of each clpass
    ax = fig.add_subplot(size[0], size[1], pos)
    ax.text(0, 0, title, horizontalalignment='left', verticalalignment='bottom', fontsize='large', fontweight='bold', transform=ax.transAxes)
    for i in range(len(data.data)):
        x, y = data.data[i].T
        ax.plot(x, y, color[int(data.labels[i])])

def main():
    global cls_algorithms, fig, test, reg_train

    generate_data()
    
    if args.algo==0:
        plot_class(origin, [2, 3], 1, 'original')
        idx = 2
        for kernel in kernels:
            transform = ClassData()
            transform.data = KernelPCA(n_components=2, kernel=kernel).fit_transform(origin.data, origin.labels)
            transform.labels = origin.labels
            plot_class(transform, [2, 3], idx, kernel)
            idx = idx + 1

    elif args.algo==1:
        plot_class(origin, [1, 2], 1, 'original')
        transform = ClassData()
        transform.data = TruncatedSVD(n_components=1).fit_transform(origin.data, origin.labels)
        transform.data = np.append(transform.data, np.zeros((len(transform.data), 1)), axis=1)
        transform.labels = origin.labels
        plot_class(transform, [1, 2], 2, 'SVD')

    elif args.algo==2:
        plot_class(origin, [1, 2], 1, 'original')
        transform = ClassData()
        transform.data = kernelFDA(origin.data, origin.labels, 3)
        transform.labels = origin.labels
        plot_class(transform, [1,2], 2, 'Kernel FDA')

    if not args.nofigure:
        plt.show()

    if args.save:
        plt.savefig('res/feature_generation_%s.png' % (feature_algorithm[args.algo]))
    
    
if __name__=='__main__':
    main()
