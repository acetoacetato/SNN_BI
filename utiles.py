import pandas as pd
import numpy as np


def load_config():
    par = np.genfromtxt("config.csv", delimiter=',')
    p = np.int(par[0])
    hn = np.int8(par[1])
    C = np.int_(par[2])
    return(p, hn, C)


def load_data_txt(fnameinp, fnameout, transpose_y=False):
    X = pd.read_csv(fnameinp, header=None)
    X = np.array(X)
    Y = pd.read_csv(fnameout, header=None)
    Y = np.array(Y)  # np.transpose(np.array(Y))
    if transpose_y:
        Y = np.transpose(Y)
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
    aux.append(RMSE)
    aux.append(R2)
    from sklearn.metrics import r2_score
    aux.append(r2_score(y.T, z.T))
    return aux


def csv_to_numpy(file_path: str) -> np.array:
    try:
        file = open(file_path, "r")
        np_array = np.loadtxt(file_path, delimiter=',')
        return np_array
    except Exception as e:
        raise Exception(
            f"Couldn't open file {file_path}. Check that route is valid")
