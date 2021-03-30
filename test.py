import numpy as np
import pandas as pd
import utiles


def test_snn(a0, w1, w2):
    z = np.dot(w1, a0)
    a1 = 1/np.exp(-z)
    a2 = np.dot(w2, a1)
    return a2


def main():
    inp = "test_x.csv"
    out = "test_y.csv"
    file_w = "pesos.npz"
    p, hn, C = utiles.load_config()
    xe, ye = utiles.load_data_txt(inp, out, True)
    w1, w2 = utiles.load_w_npy(file_w)
    y = test_snn(xe,  w1, w2)
    metricas = utiles.metrics(y, ye)
    print(metricas)


if __name__ == "__main__":
    main()
