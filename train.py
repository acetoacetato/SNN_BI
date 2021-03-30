

# Training SNN based on Pseudo-inverse

import pandas as pd
import numpy as np
import math
import random
import utiles as Utility
import math

# Calculate Pseudo-inverse


def p_inversa(a1, ye, hn, C):
    ya = np.dot(ye, a1.T)
    ai = np.dot(a1, a1.T) + np.eye(int(hn))/C
    p_inv = np.linalg.pinv(ai)
    w2 = np.dot(ya, p_inv)
    return(w2)


# Training SNN via Pseudo-inverse
def train(x, y, ocultas, C):
    n0 = x.shape[0]
    w1 = Utility.iniW(ocultas, n0)
    z = np.dot(w1, x)
    a = 1/(1+np.exp(-z))
    w2 = p_inversa(a, y, ocultas, C)
    return(w1, w2)


def main():
    inp = "train_x.csv"
    out = "train_y.csv"
    # Loading config
    p, hn, C = Utility.load_config()
    # Loading Data
    xe = Utility.csv_to_numpy(inp)
    ye = Utility.csv_to_numpy(out)
    # Training
    w1, w2 = train(xe, ye, hn, C)

    # Save weights
    Utility.save_w_npy(w1, w2)

    # Forward
    z1 = np.dot(w1, xe)
    a1 = Utility.activation(z1)

    z2 = np.dot(w2, a1)

    Utility.metrics(ye, z2, "train_metrica.csv")

    np.savetxt("train_costos.csv", np.c_[ye, z2], delimiter=",", fmt="%.6f")


if __name__ == '__main__':
    main()
