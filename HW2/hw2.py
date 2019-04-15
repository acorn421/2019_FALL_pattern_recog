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

cls_data = np.empty(shape=(0, 2))  # generated classification data
cls_labels = np.empty(shape=(0, 1)) #   labels of classification data
cls_algorithms = ["Naive Bayes", "KNN", "EM", "Logistic Regression", "SVM"]
cls_res = {}

def generate_classification_data():
    global cls_data, cls_labels

    for i in range(N):
        cls_data = np.append(cls_data, np.random.multivariate_normal(mean[i], cov[i], M), axis=0)
        cls_labels = np.append(cls_labels, np.array([i] * M))

def classification(algorithm):
    global cls_data, cls_labels, cls_res

    if algorithm == "Naive Bayes":
        model = GaussianNB()
    elif algorithm == "KNN":
        model = KNeighborsClassifier(3)
    elif algorithm == "EM":
        model = LogisticRegression()
        pass
    elif algorithm == "Logistic Regression":
        model = LogisticRegression()
    elif algorithm == "SVM":
        model = LogisticRegression()
        pass
    else:
        raise ValueError("Wrong algorithm type")

    cls_res[algorithm] = model.fit(cls_data, cls_labels).predict(cls_data)
    print(cls_res[algorithm])



def plot_results():
    global cls_data, cls_labels, cls_algorithms

    color = ('ro', 'go', 'bo')  #   colors of each class
    fig = plt.figure()
    ax = fig.add_subplot(2, (len(cls_algorithms)+1)//2, 1)
    for i in range(N):
        x, y = cls_data[M*i:M*(i+1)].T
        ax.plot(x, y, color[i], label='class'+str(i))
    ax.legend(loc=2)
        
    plt.show()

def main():
    generate_classification_data()
    for algorithm in cls_algorithms:
        classification(algorithm)
    # plot_results()
    

if __name__=='__main__':
    main()


# import code
# code.interact(local=globals())
