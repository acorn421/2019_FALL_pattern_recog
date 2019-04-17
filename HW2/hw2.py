import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

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
cls_res = {}

origin = ClassData()
train = ClassData()
test = ClassData()

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
    # origin.grid = train.grid = test.grid = np.c_[xx.ravel(), yy.ravel()]

# def generate_regression_data():


def classification(algorithm):
    global train, test

    if algorithm == "Naive Bayes":
        model = GaussianNB()
    elif algorithm == "KNN":
        model = KNeighborsClassifier(3)
    elif algorithm == "EM":
        model = LogisticRegression()
    elif algorithm == "Logistic Regression":
        model = LogisticRegression()
    elif algorithm == "SVM":
        model = LogisticRegression()
    else:
        raise ValueError("Wrong algorithm type")

    res = ClassData(data=test.data, gridx=test.gridx, gridy=test.gridy)
    res.labels = model.fit(train.data, train.labels).predict(test.data)
    res.boundary = model.predict(np.c_[res.gridx.ravel(), res.gridy.ravel()]).reshape(res.gridx.shape)

    return res


# def plot_class()

def plot_class(data, pos, boundary=False):
    global fig

    color = ('ro', 'go', 'bo')  #   colors of each class
    ax = fig.add_subplot(2, (len(cls_algorithms)+1)//2, pos)
    for i in range(len(data.data)):
        x, y = data.data[i].T
        ax.plot(x, y, color[int(data.labels[i])])
    if boundary:
        if data.boundary is None:
            raise ValueError("you must calculate boundary before plot")
        ax.contourf(data.gridx, data.gridy, data.boundary, alpha=0.4)

# def plot_origin():
#     global origin, cls_algorithms

#     color = ('ro', 'go', 'bo')  #   colors of each class
#     fig = plt.figure()
#     ax = fig.add_subplot(2, (len(cls_algorithms)+1)//2, 1)
#     for i in range(N):
#         x, y = origin.data[M*i:M*(i+1)].T
#         ax.plot(x, y, color[i], label='class'+str(i))
#     ax.legend(loc=2)
        
#     plt.show()

def main():
    generate_classification_data()
    # for algorithm in cls_algorithms:
    #     classification(algorithm)
    # plot_origin()
    plot_class(origin, 1)
    res = classification('Naive Bayes')
    plot_class(res, 2, boundary=True)
    res = classification('KNN')
    plot_class(res, 3, boundary=True)
    plt.show()
    

if __name__=='__main__':
    main()


# import code
# code.interact(local=globals())
