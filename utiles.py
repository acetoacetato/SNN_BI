import pandas as pd
import numpy as np


def load_config():
    par = np.genfromtxt("config.csv", delimiter=',')
    hn = np.int8(par[1])
    C = np.int_(par[2])
    return(hn, C)


def load_data_txt(fnameinp, fnameout):
    X = pd.read_csv(fnameinp, header=None)
    X = np.array(X)
    Y = pd.read_csv(fnameout, header=None)
    Y = np.array(Y)
    return (X, Y)


def save_w_npy(w1, w2):
    np.savez('pesos.npz', idx1=w1, idx2=w2)


def load_w_npy(file_w):
    W = np.load(file_w)
    w1 = W['idx1']
    w2 = W['idx2']
    return (w1, w2)


def iniW(next, prev):
    r = np.sqrt(6/next+prev)
    w = np.random.rand(next, prev)
    w = w * 2 * r - r
    return (w)


def metrics(y, z):
    aux = []
    e = y-z
    MAE = np.mean(np.absolute(e))
    MSE = np.mean(np.power(e, 2))
    RMSE = np.sqrt(MSE)
    R2 = 1 - (np.var(e)/np.var(y))
    aux.append(MAE)
    aux.append(MSE)
    aux.append(RMSE)
    aux.append(R2)
