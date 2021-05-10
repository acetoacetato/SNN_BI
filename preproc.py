import numpy as np
import random
import math

class Utility:

    @staticmethod
    def csv_to_numpy(file_path):
        try:
            file = open(file_path, "r")
            np_array = np.loadtxt(file_path, delimiter=',')
            return np_array
        except Exception as e:
            raise Exception(
                f"Couldn't open file {file_path}. Check that route is valid")

    @staticmethod
    def metrics(y, z, path):
        mae = abs(y - z).mean()
        mse = (np.square(y - z)).mean()
        rmse = math.sqrt(mse)
        r2 = 1-((y-z).var()/y.var())
        # print(f"MAE : {mae}")
        # print(f"MSE : {mse}")
        # print(f"RMSE : {rmse}")
        # print(f"R2 : {r2}")
        np.savetxt(path, [mae, mse, r2], delimiter=",", fmt="%.6f")
        return [mae,mse,rmse, r2]

    @staticmethod
    def load_config():
        par = np.genfromtxt("config.csv", delimiter=',')
        p = np.int(par[0])
        hn = np.int8(par[1])
        C = np.int_(par[2])
        return(p, hn, C)
    
    @staticmethod
    def save_w_npy(w1, w2):
        np.savez('pesos.npz', idx1=w1, idx2=w2)

    @staticmethod
    def load_w_npy(file_w):
        W  = np.load(file_w)
        w1  = W['idx1']
        w2  = W['idx2']
        return (w1, w2)

    @staticmethod
    def iniW(next, prev):
        r = np.sqrt(6/next+prev)
        w = np.random.rand(next, prev)
        w = w * 2 * r - r
        return w

    @staticmethod
    def iniW(hn, x0):
        r = math.sqrt(6/(hn + x0))
        matrix =  []
        for i in range(0, int(hn)):
            row = []
            for j in range(0, x0):
                row.append(random.random() * 2 * r - r)
            matrix.append(row)
        return matrix  

    @staticmethod
    def activation(z):
        return 1/ (1 + np.exp(-z))

def random_np_array(input_x, input_y):
    indx = len(input_x[0])-1
    for i in range(1, indx):
        rd_1 = random.randint(0, indx)
        rd_2 = random.randint(0, indx)
        input_x[:, [rd_1, rd_2]] = input_x[:,[rd_2, rd_1]]
        input_y[[rd_1, rd_2]] = input_y[[rd_2, rd_1]]
    return input_x, input_y

def normalize_np_array(np_array):
    a = 0.01
    b = 0.99
    if len(np_array.shape) == 2:
        normalized = []
        for row in np_array:
            max = np.amax(row)
            min = np.amin(row)
            normalized_row = list(map(lambda value: ((value-min) / (max-min)) * (b-a) + a, row))
            normalized.append(normalized_row)
        normalized = np.array(normalized)
    elif len(np_array.shape) == 1:
        max = np.amax(np_array)
        min = np.amin(np_array)
        normalized = np.array(list(map(lambda value: ((value-min) / (max-min)) * (b-a) + a, np_array)))
    return normalized

def generate_files(np_array_x : np.array, np_array_y : np.array, training_percentage : float):
    training_split_x = int(np_array_x.shape[1] * training_percentage/100)
    training_split_y = int(np_array_y.shape[0] * training_percentage/100)
    try:
        training_x = np_array_x[0:np_array_x.shape[0], 0:training_split_x]
        testing_x = np_array_x[0:np_array_x.shape[0], training_split_x+1::]
        training_y = np_array_y[0:training_split_y]
        testing_y = np_array_y[training_split_y+1::]
        np.savetxt("./train_x.csv", training_x, delimiter=",")
        np.savetxt("./train_y.csv", training_y, delimiter=",")
        np.savetxt("./test_x.csv", testing_x, delimiter=",")
        np.savetxt("./test_y.csv", testing_y, delimiter=",")
    except Exception as e:                                                            
        raise Exception("There was a problem generating training and test files.")



if __name__ == "__main__":
    input_x = Utility.csv_to_numpy("./x_input.csv")
    input_y = Utility.csv_to_numpy("./y_output.csv")
    p, hn, C = Utility.load_config()
    random_x, random_y = random_np_array(input_x, input_y)
    normalized_x = normalize_np_array(random_x)
    normalized_y = normalize_np_array(random_y)
    generate_files(normalized_x, normalized_y, p)