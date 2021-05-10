import numpy as np
import math


def csv_to_numpy(file_path):
    try:
        file = open(file_path, "r")
        np_array = np.loadtxt(file_path, delimiter=',')
        return np_array
    except Exception as e:
        raise Exception(
            f"Couldn't open file {file_path}. Check that route is valid")


def metrics(y, z, path):
    tp = {k: 0 for k in range(10)}
    fn = {k: 0 for k in range(10)}
    fp = {k: 0 for k in range(10)}

    print(y.T[0])
    for fila in range(len(z[0])):
        suma = 0
        print(len(z[fila]))
        if(np.argmax(z[fila]) == np.argmax(y.T[fila])):
            tp[np.argmax(z[fila])] = tp[np.argmax(z[fila])] + 1
        else:
            fp[np.argmax(z[fila])] = fp[np.argmax(z[fila])] + 1
            fn[np.argmax(y.T[fila])] = fn[np.argmax(y.T[fila])] + 1

        # recorriendo 10 clases
        for val in z[fila]:
            suma = suma + val

    for i in range(10):
        print(
            f'f-score clase {i}: {f_score(tp[i], fp[i], fn[i])}')

    return


def f_score(tp, fp, fn):

    precision = tp/(tp + fp+0.00000000001)
    recall = tp/(tp+fn+0.00000000001)

    f_score = 2 * (precision * recall)/(precision + recall + 0.00000000001)

    return f_score


def load_config():
    par = np.genfromtxt("config.csv", delimiter=',')
    p = np.int(par[0])
    hn = np.int8(par[1])
    mu = np.float16(par[2])
    iter = np.int(par[3])
    return(p, hn, mu, iter)


def save_w_npy(w1, w2):
    np.savez('pesos.npz', idx0=w1, idx2=w2)


def save_1_w_npy(w, capas):
    savez_dict = dict()
    for i in range(len(w)):

        savez_dict[f'idx{i}'] = w[i]

    np.savez('w_dl.npz', **savez_dict)


def load_w_npy(file_w):
    W = np.load(file_w)
    w1 = W['idx0']
    return w1


def load_1_w_npy(file_w, n_peso):
    W = np.load(file_w)
    x = np.load(file_w, mmap_mode='r')
    return W[f'idx{n_peso}']


def iniW(next, prev):
    r = np.sqrt(6/next+prev)
    w = np.random.rand(next, prev)
    w = w * 2 * r - r
    return w


def initW_snn(n0, hn, num):
    w1 = iniW(hn, n0)
    w2 = iniW(num, hn)

    return w1, w2


def softmaxEquation(scores):
    scores -= np.max(scores)
    prob = (np.exp(scores) / np.sum(np.exp(scores), axis=0))
    return prob


'''
def snn_bw_softmax(act, ye, w1, mu, lambd):
    """
    act : muestra
    ye: predicho
    w1 : pesos capa
    """

    numOfSamples = ye[0].shape[0]
    scores = np.dot(act, w1.T)
    prob = softmaxEquation(scores)
    loss = -np.log(np.max(prob)) * ye
    regLoss = (1/2)*lambd*np.sum(w1*w1)
    totalLoss = (np.sum(loss) / numOfSamples) + regLoss
    grad = ((-1 / numOfSamples) * np.dot(act.T, (ye - prob))) + (lambd*w1).T
    return totalLoss, grad
'''


def snn_bw_softmax(act, ye, w1, mu, lambd):
    """
    act : predicho
    ye: a predecir (target)
    w1 : pesos capa
    """
    numOfSamples = ye[0].shape[0]
    scores = np.dot(w1, act)
    prob = softmaxEquation(scores.T)
    loss = -np.log(np.max(prob)) * ye
    regLoss = (1/2)*lambd*np.sum(w1*w1)
    totalLoss = (np.sum(loss) / numOfSamples) + regLoss

    uno = (-1 / numOfSamples)
    dos = np.dot((ye - prob.T), act.T)
    grad = uno * dos
    grad = grad + (lambd*w1)
    return totalLoss, grad


def activation(z):
    return 1 / (1 + np.exp(-z))


def activation_derivated(a):
    return a * (1-a)


def snn_ff_list(xv, w_list):
    zv = xv
    for w in w_list:
        zv = activation(np.dot(w, zv))
    return zv


def snn_ff(xv, w1, w2):
    zv = np.dot(w1, xv)
    a1 = activation(zv)
    z2 = np.dot(w2, a1)
    a2 = activation(z2)
    return xv, a1, a2


def snn_bw(act, ye, w1, w2, mu):
    xe = act[0]
    a1 = act[1]
    a2 = act[2]

    e = (a2 - ye)

    dZ2 = np.multiply(e, activation_derivated(a2))
    dW2 = np.dot(dZ2, a1.T)

    dZ1 = np.multiply(np.dot(w2.T, dZ2), activation_derivated(a1))
    dW1 = np.dot(dZ1, xe.T)

    w1 = w1 - mu * dW1
    w2 = w2 - mu * dW2

    _, _, xv = snn_ff(xe, w1, w2)

    cost = np.mean(np.power((ye-xv), 2))

    return w1, w2, cost
