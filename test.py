import my_utility as ut
import numpy as np

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
    
    np.savetxt("test_estima.csv", np.c_[ye,y[2].T], delimiter=",", fmt="%.6f")

if __name__ == "__main__":
    main()
