import numpy as np


class Layer(object):
    """
    Classe de uma camada de uma N3.
    """
    def __init__(self, n_neurons, activation, initializer='glorotnormal'):
        """
        Classe de uma camada de uma N3.
        :param n_neurons: número de neurônimos da camada
        :param activation: modo de ativação dos neurônios da camada
        """
        self.fn_activation = activation
        self.n_neuros = n_neurons
        self.weights = None
        self.bias = None
        self.activation = None
        self.z = None
        self.dW = None
        self.dB = None
        self.initializer = initializer

    @property
    def shape(self):
        """
        Retorna formatado da camada
        :return:
        """
        return "Pesos: " + str(self.weights.shape) + " Bias: " + str(self.bias.shape)

    def backpropagation(self, erro, activation_a):
        """
        Realiza o calculo do backpropagation da camada
        :param delta: derivada do erro para o calculo do delta
        :param activation_a: ativacoes da camada anterior
        :return:
        """

        # Calculo da derivada do erro para o calculo dos deltas
        dErro = erro * self.fn_activation.back(self.activation)

        # Calculo do delta para atualização dos pesos
        self.dW = np.dot(dErro, activation_a.T) / activation_a.shape[1]

        # Calculo do delta para atualização dos bias
        self.dB = np.sum(dErro, axis=1, keepdims=True) / activation_a.shape[1]

        # Calculo da derivada do erro para proxima camada
        dErroProx = np.dot(self.weights.T, dErro)

        return dErroProx

    def init_layer(self, n_inputs):
        """
        Inicializador de uma camada usando distribuição normal dos pesos
        :param n_inputs: número de neurônios da camada anterior
        :return:
        """

        # Inicializa pesos e o bias

        if self.initializer == 'randomnormal':
            self.weights = np.random.normal(0, 0.1, (self.n_neuros, n_inputs))
            self.bias = np.random.normal(0, 0.1, (self.n_neuros, 1))

        elif self.initializer == 'glorotuniform':
            fan_in = self.n_neuros*n_inputs
            fan_out = self.n_neuros*n_inputs
            limit = np.sqrt(6 / (fan_in + fan_out))
            self.weights = np.random.uniform(-limit, limit, size=(self.n_neuros, n_inputs))
            self.bias = np.random.uniform(-limit, limit, (self.n_neuros, 1))

        return self.n_neuros

    def compute_layer(self, input):
        """
        Realiza o calculo de ativação dos neurônios da camada
        :param input: conjunto de entrada da camada
        :return: lista das ativações da camada
        """
        self.z = np.dot(self.weights, input) + self.bias
        self.activation = self.fn_activation.forward(self.z)
        return self.activation

    def update_layer(self, learning_rate):
        """
        Atualiza os pesos da camada com base no gradiente calculado
        :param learning_rate:
        :return:
        """
        self.weights -= learning_rate * self.dW
        self.bias -= learning_rate * self.dB