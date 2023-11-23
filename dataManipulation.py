# Functions for data manipulation
import numpy as np
import csv


def open_csv(file_path):
    with open(file_path, 'r') as file:
        # Create a CSV reader
        reader = csv.reader(file)
    raw_data = np.genfromtxt(file_path, delimiter=',', dtype=str, skip_header=1)
    raw_data = np.array(raw_data)
    return raw_data


def tvt_split(data):
    train_p, valid_p, test_p, data_l = 0.7, 0.15, 0.15, len(data)
    assert (train_p + valid_p + test_p) == 1, "Proportion values must add up to 100%"
    tr = int(data_l*train_p)
    va = tr + int(data_l*valid_p)
    print(tr, va)
    train = data[:tr]
    valid = data[tr:va]
    test = data[va:]
    return train, valid, test
