import my_utility as ut
import numpy as np


def cargar_capa(num_capa: int):
    w = ut.load_1_w_npy(f'pesos.npz', num_capa)
    return w


def main():
    inp = "test_x.csv"
    out = "test_y.csv"
    file_w = "pesos.npz"
    p, hn, mu, iter = ut.load_config()
    xe = ut.csv_to_numpy(inp)
    ye = ut.csv_to_numpy(out)
    numero_capa = 5
    capas = []
    for i in range(numero_capa):
        capas.append(cargar_capa(i))
    x_v = ut.snn_ff_list(xe, capas)
    x_v = ut.softmaxEquation(x_v)
    ut.metrics(ye, x_v.T, "test_metrica.csv")


'''
def main():
    inp = "test_x.csv"
    out = "test_y.csv"
    file_w = "pesos.npz"
    # Load Config
    p, hn, mu, iter = ut.load_config()
    # Load Data
    xe = ut.csv_to_numpy(inp)
    ye = ut.csv_to_numpy(out)
    # Load weights
    w1, w2 = ut.load_w_npy(file_w)

    # Forward
    y = ut.snn_ff(xe, w1, w2)
    # Make Metrics
    metricas = ut.metrics(ye, y[2], "test_metrica.csv")

    np.savetxt("test_estima.csv", np.c_[ye, y[2].T], delimiter=",", fmt="%.6f")
'''

if __name__ == "__main__":
    main()
