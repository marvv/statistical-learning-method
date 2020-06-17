import numpy as np
import time


def load_data(path):
    data = np.genfromtxt(path, delimiter=',')
    labels = []
    for d in data:
        if d[0] >= 5:
            labels.append(1)
        else:
            labels.append(-1)
    return np.asmatrix(data), np.asmatrix(labels).T


def calc_accuracy(dataMat, labelMat, w, b):
    m, _ = np.shape(dataMat)
    errCnt = 0
    for i in range(m):
        xi = dataMat[i]
        yi = labelMat[i]
        if -1 * yi * (w * xi.T + b) >= 0:
            errCnt += 1
    return 1 - (errCnt / m)


def train(dataMat, labelMat, epoch, eta):
    m, n = np.shape(dataMat)
    w = np.zeros((1, n))
    b = 0
    epoch = 100
    eta = 0.0001

    for ep in range(epoch):
        for i in range(m):
            xi = dataMat[i]
            yi = labelMat[i]
            if -1 * yi * (w * xi.T + b) >= 0:
                w = w + eta * yi * xi
                b = b + eta * yi
        print('round %d, %d' % (ep, epoch))
    return w, b


if __name__ == "__main__":
    dataMat, labelMat = load_data('./mnist_test.csv')
    testData, testLabel = load_data('./mnist_test.csv')

    start = time.time()
    w, b = train(dataMat, labelMat, 100, 0.0001)
    accruRate = calc_accuracy(testData, testLabel, w, b)
    end = time.time()
    print('accuracy rate is:', accruRate)
    print('time span:', end - start)
