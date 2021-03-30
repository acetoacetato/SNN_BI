

# Training SNN based on Pseudo-inverse

import pandas as pd
import numpy as np
import math
import random
import utiles


class ut:
    def iniW(hn, x0):
        pesos = np.zeros((hn, x0))
        n0 = hn
        n1 = 1
        r = math.sqrt(6/(n1+n0))
        for i in range(len(pesos)):
            for j in range(len(pesos[0])):
                w_i = random.uniform(n1, n0)
                pesos[i][j] = w_i
        return pesos

    def load_data_txt(inp, out):
        df_inp = np.genfromtxt(inp, dtype=None, delimiter=',')
        df_out = np.genfromtxt(out, dtype=None, delimiter=',')
        return (df_inp, df_out)

    def save_w_npy(w1, w2):
        np.savetxt('w1.npy', w1)
        np.savetxt('w2.npy', w2)

# Calculate Pseudo-inverse


def p_inversa(a1, ye, hn, C):
    ya = np.dot(ye, a1.T)
    ai = np.dot(a1, a1.T) + np.eye(hn)/C
    p_inv = np.linalg.pinv(ai)
    w2 = np.dot(ya, p_inv)
    return(w2)


# Training SNN via Pseudo-inverse
def train_snn(xe, ye, hn, C):
    n0 = xe.shape[0]
    w1 = utiles.iniW(hn, n0)
    z = np.dot(w1, xe)
    a1 = 1/(1+np.exp(-z))
    w2 = p_inversa(a1, ye, hn, C)
    return(w1, w2)


def main():
    inp = "train_x.csv"
    out = "train_y.csv"
    p, hn, C = utiles.load_config()
    xe, ye = utiles.load_data_txt(inp, out)
    w1, w2 = train_snn(xe, ye, hn, C)
    utiles.save_w_npy(w1, w2)


if __name__ == '__main__':
    main()
