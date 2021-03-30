import numpy as np
import utiles as Utility
import random


def random_np_array(input_x, input_y):
    indx = len(input_x[0])-1
    for i in range(1, indx):
        rd_1 = random.randint(0, indx)
        rd_2 = random.randint(0, indx)
        input_x[:, [rd_1, rd_2]] = input_x[:, [rd_2, rd_1]]
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
            normalized_row = list(
                map(lambda value: ((value-min) / (max-min)) * (b-a) + a, row))
            normalized.append(normalized_row)
        normalized = np.array(normalized)
    elif len(np_array.shape) == 1:
        max = np.amax(np_array)
        min = np.amin(np_array)
        normalized = np.array(
            list(map(lambda value: ((value-min) / (max-min)) * (b-a) + a, np_array)))
    return normalized


def generate_files(np_array_x: np.array, np_array_y: np.array, training_percentage: float):
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
        raise Exception(
            "There was a problem generating training and test files.")


if __name__ == "__main__":
    input_x = Utility.csv_to_numpy("./x_input.csv")
    input_y = Utility.csv_to_numpy("./y_output.csv")
    p, hn, C = Utility.load_config()
    random_x, random_y = random_np_array(input_x, input_y)
    normalized_x = normalize_np_array(random_x)
    normalized_y = normalize_np_array(random_y)
    generate_files(normalized_x, normalized_y, p)
