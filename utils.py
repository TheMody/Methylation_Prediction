import numpy as np

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
    for i in range(folds):
        x_train = np.concatenate([x_folds[j] for j in range(folds) if j != i])
        y_train = np.concatenate([y_folds[j] for j in range(folds) if j != i])
        x_test = x_folds[i]
        y_test = y_folds[i]
        cls.fit(x_train, y_train)
        accs.append(np.sum(np.round(cls.predict(x_test)) == y_test)/len(y_test))
    return np.mean(accs), np.var(accs)