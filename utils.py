from math import floor
import numpy as np
def extract_features(img_arr,v):
    feat = [0 for i in range(int((28/v)**2))]
    for i in range(28):
        for j in range(0,28,v):
            ind = int( floor(i/v)*(28/v)+j/v)
            feat[ind] += sum(img_arr[i,j:j+v])
            if feat[ind] == 0:
                feat[ind] == 0.00000001
    feat = [round(x/(v**2),2) for x in feat]
    return feat
def extract_y_trains(labels):
    y_trains = []
    for i in [0,1,2,3,4,5,6,7,8,9]:
        y_train = []
        for label in labels:
            if label ==i:
                y_train.append(1)
            else:
                y_train.append(0)
        y_trains.append(np.reshape(y_train,[-1,1]))
    return y_trains

def sigmoid(x):
    return 1 / (1 + np.e ** (-x))


def logistic_cost_function(w, x_train, y_train):
    N = np.shape(x_train)[0]
    sigm = sigmoid(x_train @ w)
    log_func = -1*sum([y_train[i]*np.log(sigm[i])+(1-y_train[i])*np.log(1-sigm[i]) for i in range(N)])/N
    grad = x_train.transpose() @ (sigm - y_train) / N
    return log_func,grad


def stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch):
    w = w0
    M = int(len(x_train) / mini_batch)
    log_values = []
    packs = [(x_train[i:(mini_batch + i)], y_train[i:(mini_batch + i)]) for i in range(0, len(x_train), mini_batch)]
    for k in range(epochs):
        print("epoch ",k)
        for m in range(M):
            logval, grad = obj_fun(w, packs[m][0], packs[m][1])
            w = w - eta * grad
        logval, grad = obj_fun(w, x_train, y_train)
        log_values.append(logval)
    return w, np.reshape(log_values,[epochs,])

def regularized_logistic_cost_function(w, x_train, y_train, regularization_lambda):
    log, grad = logistic_cost_function(w, x_train, y_train)
    log += regularization_lambda / 2 * np.linalg.norm(w[1:],2) ** 2
    grad[1:] = np.add(grad[1:], regularization_lambda * w[1:])
    return log, grad


def prediction(x, w, theta):
    return sigmoid(x @ w) >= theta


def f_measure(y_true, y_pred):
    TP = 0
    FNP = 0
    vals = y_true+y_pred
    for i in vals:
        if i == 2:
            TP+=1
        elif i == 1:
            FNP += 1
    return 2*TP/(2*TP+FNP)


def model_selection(x_train, y_train, x_val, y_val, w0, epochs, eta, mini_batch, lambdas, thetas):
    F = []
    results = []
    for l in lambdas:
        print(l)
        w = stochastic_gradient_descent(lambda w, x, y: regularized_logistic_cost_function(w, x, y, l),
                                           x_train, y_train, w0, epochs, eta, mini_batch)[0]
        F_row = []
        for t in thetas:
            results.append((l, t, w))
            F_row.append(f_measure(y_val, prediction(x_val, w, t)))
        F.append(F_row)
    regularization_lambda, theta, w = results[np.argmax(F)]
    return regularization_lambda, theta, list(w), max(F)