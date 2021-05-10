import numpy as np
import pandas as pd
import my_utility as ut


def cargar_capa(num_capa: int):
    w = ut.load_1_w_npy(f'w_dl.npz', num_capa)
    return w


def train_ae(hnode, neur_capa_ant, mu, maxiter, x, pinv):
    w1, _ = ut.initW_snn(neur_capa_ant, x.shape[0], neur_capa_ant)
    for iter in range(1, maxiter):
        w2 = ut.pinv_ae(x, w1, pinv)
        w1, cost = ut.backward_ae(x, w1, w2, mu)
    return w1


def train_sm(xe, ye, mu, iter, numero_capa, lambd):
    capas = []
    for i in range(numero_capa):
        capas.append(cargar_capa(i))

    x_v = ut.snn_ff_list(xe, capas)
    w1, _ = ut.initW_snn(x_v.shape[0], ye.shape[0], ye.shape[0])
    mse = []
    f = open('costo_softmax.csv', 'w')
    for i in range(int(iter)):

        grad, cost = ut.snn_bw_softmax(x_v, ye, w1, lambd)
        w1 = w1 - mu * grad
        if(i % 10 == 0):
            print(f'iter={i} cost={cost}')
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
    pinv = int(f_sae.readline())
    maxiter = int(f_sae.readline())
    capas = [int(f) for f in f_sae.readlines()]

    maxiter_sm = int(f_sm.readline())
    mu_sm = float(f_sm.readline())
    lambd_sm = float(f_sm.readline())

    xe = ut.csv_to_numpy('train_x.csv')
    ye = ut.csv_to_numpy('train_y.csv')

    for i in range(len(capas)):
        train_ae(i, capas[i], mu, maxiter, xe, pinv)
    train_sm(xe, ye, mu_sm, maxiter_sm, len(capas), lambd_sm)


if __name__ == '__main__':
    main()
