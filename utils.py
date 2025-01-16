import numpy as np
from matplotlib import pyplot as plt
from config import *
import pickle
def crossvalidate(cls, x,y, folds = 5):
    #shuffle data
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    #split data into folds
    x_folds = np.array_split(x, folds)
    y_folds = np.array_split(y, folds)
    accs = []
    loss = []
    for i in range(folds):
        x_train = np.concatenate([x_folds[j] for j in range(folds) if j != i])
        y_train = np.concatenate([y_folds[j] for j in range(folds) if j != i])
        x_test = x_folds[i]
        y_test = y_folds[i]
        cls.fit(x_train, y_train)
        l1error = np.abs(cls.predict(x_test)-y_test)
        loss.append(np.mean(l1error))
        accs.append(np.sum(np.round(cls.predict(x_test)) == y_test)/len(y_test))
    return np.mean(loss), np.var(loss), np.mean(accs), np.var(accs)

def analyze_predictions(y,y_pred):
    accuracy_per_class = np.zeros(num_classes)
    counts_per_class = np.zeros(num_classes)
    for i in range(len(y)):
        accuracy_per_class[y[i]] += y_pred[i] == y[i]
        counts_per_class[y[i]] += 1.0
    accuracy_per_class = accuracy_per_class/counts_per_class
    overall_accuracy = np.sum(y == y_pred)/len(y)
    return accuracy_per_class, overall_accuracy

def analyze_difference_in_predictions(y,y_pred, y_pred_2, save_name ="/accuracy_diff" ):
    a_1, oa1 = analyze_predictions(y,y_pred)
    a_2, oa2 = analyze_predictions(y,y_pred_2)
    plt.bar(range(num_classes), a_1, color = "red")
    plt.bar(range(num_classes), a_2, color = "blue")
   # plt.bar(range(num_classes), a_1 - a_2, color = "green")
    plt.ylabel("accuracy and their differences")
    plt.xlabel("class")
    plt.legend(["accuracy deep", "accuracy lin", "a deep - a lin"])
    plt.savefig(folder+save_name+".png")
    plt.close()
    with open(folder+save_name+".data", "wb") as f:
        pickle.dump((a_1,a_2), f)
    

