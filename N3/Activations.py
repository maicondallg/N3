import pickle

import numpy as np


# Funções de atitação com ativação e derivada da ativação

class Softmax:
    @staticmethod
    def forward(x):
        return np.exp(x)/np.sum(np.exp(x), axis=0)

    @staticmethod
    def back(x):
        s = np.exp(x)/np.sum(np.exp(x), axis=0)
        return s*(1-s)


class Tanh:
    @staticmethod
    def forward(x):
        return (np.exp(x) - np.exp(-1)) / (np.exp(x) + np.exp(-x))

    @staticmethod
    def back(x):
        return 1 - x ** 2


class Sigmoid:
    @staticmethod
    def forward(x):
        try:
            return 1/(1 + np.exp(-x))
        except RuntimeWarning:
            print('Oi')

    @staticmethod
    def back(x):
        return x * (1 - x)


class ReLU:
    @staticmethod
    def forward(x):
        return np.where(x < 0, 0.0, x)

    @staticmethod
    def back(x):
        return np.where(x < 0, 0.0, 1.0)

