import pandas as pd
import numpy as np
import math
import random
import my_utility as ut


def p_inversa(a1, ye, hn, C):
    ya = np.dot(ye, a1.T)
    ai = np.dot(a1, a1.T) + np.eye(int(hn))/C
    p_inv = np.linalg.pinv(ai)
    w2 = np.dot(ya, p_inv)
    return(w2)


def train_snn(xe, ye, nh, mu, iter):
    w1, w2 = ut.initW_snn(xe.shape[0], nh, 1)
    mse = []
    for i in range(int(iter)):
        act = ut.snn_ff(xe, w1, w2)
        w1, w2, cost = ut.snn_bw(act, ye, w1, w2, mu)
        if (i%200 == 0):
            print(f"Iteracion : {i} : Costo : {cost}")
        mse.append(cost)
    return w1, w2, mse


def main():
    inp = "train_x.csv"
    out = "train_y.csv"
    # Loading config
    _, hn, mu, iter = ut.load_config()

    # Loading Data
    xe = ut.csv_to_numpy(inp)
    ye = ut.csv_to_numpy(out)

    # Training
    w1, w2, mse = train_snn(xe, ye, hn, mu, iter)

    # Save weights
    ut.save_w_npy(w1, w2)


    np.savetxt("train_costo.csv", mse, delimiter=",", fmt="%.6f")


if __name__ == '__main__':
    main()
