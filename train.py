import numpy as np
import pandas as pd

config = pd.read_csv('./config.csv')
print(config.head())
# % training, # nodos ocultos, penalidad P-Inversa

por_training = config / 100

print(por_training)
