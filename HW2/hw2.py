import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

# K = 8
N = 3   # number of classes
M = 100 # number of data for each class
mean = [[1, 1], [7, 7], [15, 1]]    #   means of classification data
cov = [[[12, 0], [0, 1]], [[8, 3], [3, 2]], [[2, 0], [0, 2]]]   # covariance of classification data

class ClassData:
    def __init__(self, data=np.empty(shape=(0, 2)), labels=np.empty(shape=(0, 1)), pair=np.empty(shape=(0, 3)), gridx=None, gridy=None):
        self.data = data
        self.labels = labels
        self.pair = pair
        self.gridx = gridx
        self.gridy = gridy
        self.boundary = None

cls_algorithms = ["Naive Bayes", "KNN", "EM", "Logistic Regression", "SVM"]

origin = ClassData()
train = ClassData()
test = ClassData()

class RegData:
    def __init__(self, x=np.empty(shape=(0, 1)), y=np.empty(shape=(0, 1)), predict=np.empty(shape=(0, 1))):
        self.x = x
        self.y = y
        self.predict = predict

reg_origin = RegData()
reg_train = RegData()
reg_test = RegData()

fig = plt.figure()

def generate_classification_data():
    global origin, train, test

    for i in range(N):
        origin.data = np.append(origin.data, np.random.multivariate_normal(mean[i], cov[i], M), axis=0)
        origin.labels = np.append(origin.labels, np.array([i] * M))
    origin.pair = np.append(origin.data, origin.labels.reshape((len(origin.labels), 1)), axis = 1)
    train.pair, test.pair = np.split(np.random.permutation(origin.pair), [int(N*M*0.7)])
    train.data, train.labels = np.split(train.pair, [2], axis=1)
    test.data, test.labels = np.split(test.pair, [2], axis=1)
    x_min, x_max = origin.data[:, 0].min() - 1, origin.data[:, 0].max() + 1
    y_min, y_max = origin.data[:, 1].min() - 1, origin.data[:, 1].max() + 1
    origin.gridx, origin.gridy = train.gridx, train.gridy = test.gridx, test.gridy =  np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

def generate_regression_data():
    global reg_origin, reg_train, reg_test
    x, y = make_regression(n_samples=M, n_features=1, noise=40.0, shuffle=False, bias=10.0)
    data = np.append(x, y.reshape(len(y), 1), axis=1)
    data = data[data[:,0].argsort(axis=0)]
    reg_origin.x, reg_origin.y = data.T
    reg_origin.x = reg_origin.x.reshape(len(reg_origin.x), 1)
    reg_origin.y = reg_origin.y.reshape(len(reg_origin.y), 1)
    reg_train.x, reg_test.x = np.split(reg_origin.x, [int(M*0.7)])
    reg_train.y, reg_test.y = np.split(reg_origin.y, [int(M*0.7)])

def fit(algorithm):
    global train

    if algorithm == "Naive Bayes":
        model = GaussianNB()
    elif algorithm == "KNN":
        model = KNeighborsClassifier(3)
    elif algorithm == "EM":
        model = GaussianMixture(3)
    elif algorithm == "Logistic Regression":
        model = LogisticRegression(multi_class='auto', solver='lbfgs')
    elif algorithm == "SVM":
        model = SVC(kernel="linear")
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

def regression(model):
    global reg_train

    res = RegData(x=reg_test.x, y=reg_test.y)
    model.fit(reg_train.x, reg_train.y)
    res.predict = model.predict(res.x)

    return (res, model)

def plot_class(data, pos, title=None, model=None, boundary=False):
    global fig

    color = ('ro', 'go', 'bo')  #   colors of each clpass
    ax = fig.add_subplot(6, 3, pos)
    # ax.set_title(title)
    ax.text(0, 0, title, horizontalalignment='left', verticalalignment='bottom', fontsize='large', fontweight='bold', transform=ax.transAxes)
    # ax.axis('equal')
    ax.set_xlim(data.gridx[0][0], data.gridx[0][-1])e
    ax.set_ylim(data.gridy[0][0], data.gridy[-1][0])
    for i in range(len(data.data)):
        x, y = data.data[i].T
        ax.plot(x, y, color[int(data.labels[i])])
    if model is not None and model.algorithm=='SVM':
        boundary = False
        for i in range(len(model.intercept_)):
            ax.plot(data.gridx[0], -data.gridx[0]*(model.coef_[i][0]/model.coef_[i][1])-model.intercept_[i]/model.coef_[i][1], 'xkcd:black')
        if len(data.data) > N*M*0.3:
            for sv in model.support_vectors_:
                x, y = sv
                ax.plot(x, y, 'ko')
    if boundary:
        if data.boundary is None:
            raise ValueError("you must calculate boundary before plot")
        ax.contourf(data.gridx, data.gridy, data.boundary, alpha=0.2)

def plot_reg(data, pos, title, model=None):
    global reg_origin, reg_train

    ax = fig.add_subplot(6, 3, pos)
    ax.text(0, 0, title, horizontalalignment='left', verticalalignment='bottom', fontsize='large', fontweight='bold', transform=ax.transAxes)
    if model is not None:
        ax.scatter(reg_train.x, reg_train.y, c='g')
        ax.scatter(data.x, data.y, c='r')
        ax.plot(reg_origin.x, model.predict(reg_origin.x), c='k')
    else:
        ax.scatter(data.x, data.y, c='b')

def main():
    global cls_algorithms, fig, test, reg_train
    
    generate_regression_data()
    plot_reg(reg_origin, 1, 'original regression data')
    res, model = regression(LinearRegression())
    plot_reg(res, 2, 'Linear regression', model=model)

    generate_classification_data()
    plot_class(origin, 3, 'original classification data')
    for idx in range(5):
        sv = ds = None
        train_res, model = fit(cls_algorithms[idx])
        test_predict = classification(model)
        test_res = test
        test_res.boundary = test_predict.boundary
        plot_class(train_res, (idx+1)*3+1, cls_algorithms[idx]+' training data', model, boundary=True)
        plot_class(test_res, (idx+1)*3+2, cls_algorithms[idx]+' test data', model, boundary=True)
        plot_class(test_predict, (idx+1)*3+3, cls_algorithms[idx]+' prediction results', model, boundary=False)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.show()
    

if __name__=='__main__':
    main()
