import numpy as np
import pandas as pd
import utiles as Utility


def test_snn(a0, w1, w2):
    z = np.dot(w1, a0)
    a1 = 1/(1+np.exp(-z))
    a2 = np.dot(w2, a1)
    return a2


def main():
    inp = "test_x.csv"
    out = "test_y.csv"
    file_w = "pesos.npz"
    # Load Config
    p, hn, C = Utility.load_config()
    # Load Data
    xe = Utility.csv_to_numpy(inp)
    ye = Utility.csv_to_numpy(out)
    # Load weights
    w1, w2 = Utility.load_w_npy(file_w)
    # Forward
    y = test_snn(xe,  w1, w2)
    # Make Metrics
    metricas = Utility.metrics(y, ye, "test_metrica.csv")
    np.savetxt("test_costo.csv", np.c_[ye, y], delimiter=",", fmt="%.6f")


if __name__ == "__main__":
    main()
