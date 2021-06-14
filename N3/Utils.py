import numpy as np


def to_categorical(y):
    num_classes = np.unique(y).shape[0]

    num_instances = y.shape[0]
    categorical = np.zeros((num_instances, num_classes))
    categorical[np.arange(num_instances), y] = 1
    return categorical