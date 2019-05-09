import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(prog='hw3', description='Homework3 of Pattern Recognition Spring 2019')
parser.add_argument('algo', type=int, help="Non-linear classification algorithm{'Multi-Layerd Perceptron' : 0, 'RBF Network' : 1, 'Kernel SVM' : 2, 'Random Forest' : 3)", default=0)
parser.add_argument('-o', '--opt', type=int, help='option for classfication algorithm(RBF Network : number of clusters, Kernel SVM : type of kernel functions, Random Forest : number of decision trees')
args = parser.parse_args()

N = 9   # number of classes

default_opt = [0, 5, 0, 10]
kernels = ['rbf', 'sigmoid', 'linear', 'poly']

class ClassData:
    def __init__(self, data=np.empty(shape=(0, 2)), labels=np.empty(shape=(0, 1)), pair=np.empty(shape=(0, 3))):
        self.data = data
        self.labels = labels
        self.pair = pair

cls_algorithms = ["Multi-Layerd Perceptron", "RBF Network", "Kernel SVM", "Random Forest"]

origin = ClassData()
train = ClassData()
test = ClassData()

fig = plt.figure()

def generate_classification_data():
    global origin, train, test

    mnist = fetch_mldata('MNIST original')
    origin.data = mnist.data
    origin.labels = mnist.target

    train.data, test.data, train.labels, test.labels = train_test_split(origin.data, origin.labels, test_size=0.3)

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

    res = ClassData(data=train.data, labels=train.labels)
    model.fit(train.data, train.labels)
    model.algorithm = algorithm

    return (res, model)

def classification(model):
    global train, test

    res = ClassData(data=test.data)
    res.labels = model.predict(test.data)

    return res

def main():
    global cls_algorithms, fig, test, reg_train

    if not args.opt:
        args.opt = default_opt[args.algo]
    
    generate_classification_data()

    train_res, model = fit(cls_algorithms[args.algo], args.opt)
    test_predict = classification(model)

    print("Classification report for classifier \n%s:\n%s\n"
      % (model, metrics.classification_report(test.labels, test_predict.labels)))

    print("Confusion matrix:\n%s" % metrics.confusion_matrix(test.labels, test_predict.labels))

if __name__=='__main__':
    main()
