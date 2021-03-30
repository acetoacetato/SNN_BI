import pandas as pd
import numpy as np
import math
import random


def csv_to_numpy(file_path):
    try:
        file = open(file_path, "r")
        np_array = np.loadtxt(file_path, delimiter=',')
        return np_array
    except Exception as e:
        raise Exception(
            f"Couldn't open file {file_path}. Check that route is valid")


def metrics(y, z, path):
    mae = abs(y - z).mean()
    mse = (np.square(y - z)).mean()
    rmse = math.sqrt(mse)
    r2 = 1-((y-z).var()/y.var())

    np.savetxt(path, [mae, mse, r2], delimiter=",", fmt="%.6f")
    return [mae, mse, rmse, r2]


def load_config():
    par = np.genfromtxt("config.csv", delimiter=',')
    p = np.int(par[0])
    hn = np.int8(par[1])
    C = np.int_(par[2])
    return(p, hn, C)


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
    return w


def iniW(hn, x0):
    r = math.sqrt(6/(hn + x0))
    matrix = []
    for i in range(0, int(hn)):
        row = []
        for j in range(0, x0):
            row.append(random.random() * 2 * r - r)
        matrix.append(row)
    return matrix


def activation(z):
    return 1 / (1 + np.exp(-z))
