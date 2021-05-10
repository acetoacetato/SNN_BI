import numpy as np
import pandas as pd
import my_utility as ut


def cargar_capa(num_capa: int):
    w = ut.load_1_w_npy(f'pesos.npz', num_capa)
    return w


def train_snn(xe, ye, nh, mu, iter):
    w1, w2 = ut.initW_snn(xe.shape[0], nh, 1)
    mse = []
    for i in range(int(iter)):
        act = ut.snn_ff(xe, w1, w2)
        w1, w2, cost = ut.snn_bw(act, ye, w1, w2, mu)
        if (i % 200 == 0):
            print(f"Iteracion : {i} : Costo : {cost}")
        mse.append(cost)
    return w1, w2, mse


def main():
    with open('capa.txt', 'r') as info_capa:
        x_v = ut.csv_to_numpy('train_x.csv')
        _, hn, mu, iter = ut.load_config()

        # contiene el número de la capa nueva
        numero_capa = int(info_capa.readline())
        neur_capa_ant = int(info_capa.readline())
        # Cargamos todas las capas anteriores
        capas = []
        for i in range(numero_capa):
            capas.append(cargar_capa(i))

        if len(capas) == 0:
            # Creamos la primera capa
            print('nada')
        else:
            # Generamos la salida para poder entrenar el autoencoder nuevo
            xe = ut.snn_ff_list(x_v, capas)
            hn = neur_capa_ant
            w1, w2, mse = train_snn(xe, xe, hn, mu, iter)
            capas.append(w1)
            ut.save_1_w_npy(capas, numero_capa)


if __name__ == '__main__':
    main()
