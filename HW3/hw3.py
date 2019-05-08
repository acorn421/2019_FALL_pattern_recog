import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


parser = argparse.ArgumentParser(prog='SimpleInc', description='Fuzz testing using basic increment')
parser.add_argument('algo', type=int, help="Non-linear classification algorithm{'Multi-Layerd Perceptron' : 0, 'RBF Network' : 1, 'Kernel SVM' : 2, 'Random Forest' : 3)", default=0)
parser.add_argument('')
parser.add_argument('-o', '--opt', type=int, help='option for classfication algorithm(RBF Network : number of clusters, Kernel SVM : type of kernel functions, Random Forest : number of decision trees')
args = parser.parse_args()

N = 2   # number of classes
M = 200 # number of data for each class
mean = [[2, 2], [5, 5], [8, 8]]    #   means of classification data
cov = [[[2, 0], [0, 2]], [[4, -3], [-3, 4]], [[2, 0], [0, 2]]]   # covariance of classification data

default_opt = [0, 5, 0, 10]
kernels = ['rbf', 'sigmoid', 'linear', 'poly']

class ClassData:
    def __init__(self, data=np.empty(shape=(0, 2)), labels=np.empty(shape=(0, 1)), pair=np.empty(shape=(0, 3)), gridx=None, gridy=None):
        self.data = data
        self.labels = labels
        self.pair = pair
        self.gridx = gridx
        self.gridy = gridy
        self.boundary = None

cls_algorithms = ["Multi-Layerd Perceptron", "RBF Network", "Kernel SVM", "Random Forest"]

origin = ClassData()
train = ClassData()
test = ClassData()

fig = plt.figure()

def generate_classification_data():
    global origin, train, test

    origin.data = np.append(origin.data, np.random.multivariate_normal(mean[0], cov[0], int(M/2)), axis=0)
    origin.data = np.append(origin.data, np.random.multivariate_normal(mean[2], cov[2], int(M/2)), axis=0)
    origin.labels = np.append(origin.labels, np.array([0] * M))
    origin.data = np.append(origin.data, np.random.multivariate_normal(mean[1], cov[1], M), axis=0)
    origin.labels = np.append(origin.labels, np.array([1] * M))

    origin.pair = np.append(origin.data, origin.labels.reshape((len(origin.labels), 1)), axis = 1)
    train.pair, test.pair = np.split(np.random.permutation(origin.pair), [int(N*M*0.7)])
    train.data, train.labels = np.split(train.pair, [2], axis=1)
    test.data, test.labels = np.split(test.pair, [2], axis=1)
    x_min, x_max = origin.data[:, 0].min() - 1, origin.data[:, 0].max() + 1
    y_min, y_max = origin.data[:, 1].min() - 1, origin.data[:, 1].max() + 1
    origin.gridx, origin.gridy = train.gridx, train.gridy = test.gridx, test.gridy =  np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

def fit(algorithm, opt):
    global train

    if algorithm == "Multi-Layerd Perceptron":
        model = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(100, ))
    elif algorithm == "RBF Network":
        # model = KNeighborsClassifier(3)
        pass
    elif algorithm == "Kernel SVM":
        model = SVC(kernel=kernels[opt])
    elif algorithm == "Random Forest":
        model = RandomForestClassifier(n_estimators=opt, max_depth=2)
    else:
        raise ValueError("Wrong algorithm type")

    res = ClassData(data=train.data, labels=train.labels, gridx=train.gridx, gridy=train.gridy)
    model.fit(train.data, train.labels.ravel())
    model.algorithm = algorithm
    res.boundary = model.predict(np.c_[res.gridx.ravel(), res.gridy.ravel()]).reshape(res.gridx.shape)

    return (res, model)

def classification(model):
    global train, test

    res = ClassData(data=test.data, gridx=test.gridx, gridy=test.gridy)
    res.labels = model.predict(test.data)
    res.boundary = model.predict(np.c_[res.gridx.ravel(), res.gridy.ravel()]).reshape(res.gridx.shape)

    return res

def plot_class(data, pos, title=None, model=None, boundary=False):
    global fig

    color = ('ro', 'go', 'bo')  #   colors of each clpass
    ax = fig.add_subplot(2, 3, pos)
    ax.text(0, 0, title, horizontalalignment='left', verticalalignment='bottom', fontsize='large', fontweight='bold', transform=ax.transAxes)
    # ax.axis('equal')
    ax.set_xlim(data.gridx[0][0], data.gridx[0][-1])
    ax.set_ylim(data.gridy[0][0], data.gridy[-1][0])
    for i in range(len(data.data)):
        x, y = data.data[i].T
        ax.plot(x, y, color[int(data.labels[i])])
    if model is not None and model.algorithm=='Kernel SVM':
        # boundary = False
        if len(data.data) > N*M*0.3:
            for sv in model.support_vectors_:
                x, y = sv
                ax.plot(x, y, 'ko')
    if boundary:
        if data.boundary is None:
            raise ValueError("you must calculate boundary before plot")
        ax.contourf(data.gridx, data.gridy, data.boundary, alpha=0.2)

def main():
    global cls_algorithms, fig, test, reg_train

    if not args.opt:
        args.opt = default_opt[args.algo]
    
    generate_classification_data()
    plot_class(origin, 2, 'original classification data')

    sv = ds = None
    train_res, model = fit(cls_algorithms[args.algo], args.opt)
    test_predict = classification(model)
    test_res = test
    test_res.boundary = test_predict.boundary
    plot_class(train_res, 4, cls_algorithms[args.algo]+' training data', model, boundary=True)
    plot_class(test_res, 5, cls_algorithms[args.algo]+' test data', model, boundary=True)
    plot_class(test_predict, 6, cls_algorithms[args.algo]+' prediction results', model, boundary=True)

    # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.show()
    

if __name__=='__main__':
    main()
