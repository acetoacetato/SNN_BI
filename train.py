import numpy as np
import pandas as pd
import my_utility as ut


def cargar_capa(num_capa: int):
    w = ut.load_1_w_npy(f'w_dl.npz', num_capa)
    return w


def train_snn(xe, ye, nh, mu, iter):
    w1, w2 = ut.initW_snn(xe.shape[0], nh, ye.shape[0])
    mse = []
    for i in range(int(iter)):
        act = ut.snn_ff(xe, w1, w2)
        w1, w2, cost = ut.snn_bw(act, ye, w1, w2, mu)
        # if (i % 20 == 0):
        #    print(f"Iteracion : {i} : Costo : {cost}")
        mse.append(cost)
    return w1, w2, mse


def train_ae(numero_capa, neur_capa_ant, mu, maxiter, x_v):
    # Cargamos todas las capas anteriores
    capas = []
    for i in range(numero_capa):
        capas.append(cargar_capa(i))

    if len(capas) == 0:
        # Creamos la primera capa
        xe = x_v
    else:
        # Generamos la salida para poder entrenar el autoencoder nuevo
        xe = ut.snn_ff_list(x_v, capas)
    hn = neur_capa_ant
    w1, w2, mse = train_snn(xe, xe, hn, mu, maxiter)
    capas.append(w1)
    ut.save_1_w_npy(capas, numero_capa)
    return capas


def train_sm(xe, ye, mu, iter, numero_capa, lambd):
    capas = []
    for i in range(numero_capa):
        capas.append(cargar_capa(i))

    x_v = ut.snn_ff_list(xe, capas)
    w1, _ = ut.initW_snn(x_v.shape[0], ye.shape[0], ye.shape[0])
    mse = []
    f = open('costo_softmax.csv', 'w')
    for i in range(int(iter)):

        cost, grad = ut.snn_bw_softmax(x_v, ye, w1, mu, lambd)
        w1 = w1 - mu * grad
        f.write(f'{cost}\n')
        mse.append(cost)
    capas.append(w1)
    ut.save_1_w_npy(capas, numero_capa)
    f.close()
    return capas


def main():
    f_sae = open('param_sae.csv')
    f_sm = open('param_softmax.csv')

    _ = f_sae.readline()
    mu = float(f_sae.readline())
    _ = f_sae.readline()
    maxiter = int(f_sae.readline())
    capas = [int(f) for f in f_sae.readlines()]

    maxiter_sm = int(f_sm.readline())
    mu_sm = float(f_sm.readline())
    lambd_sm = float(f_sm.readline())

    xe = ut.csv_to_numpy('train_x.csv')
    ye = ut.csv_to_numpy('train_y.csv')

    for i in range(len(capas)):
        train_ae(i, capas[i], mu, maxiter, xe)
    train_sm(xe, ye, mu_sm, maxiter_sm, len(capas), lambd_sm)


if __name__ == '__main__':
    main()
