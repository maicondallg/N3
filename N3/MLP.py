import numpy as np
from sklearn.metrics import accuracy_score, f1_score, auc, balanced_accuracy_score, roc_curve, confusion_matrix


class MLP(object):
    def __init__(self, layers, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.layers = layers

    def _init_layers(self, n_features):
        """
        Função de inicialização das camadas
        :param n_features: número de neúronios da camada anterior
        :return: None
        """
        n_inputs = n_features
        for layer in self.layers:
            n_inputs = layer.init_layer(n_inputs)

    def fit(self, X, y, epochs=1000, batch_size=None, verbose=True, validation=None):
        """
        Função de treino da rede N3
        :param X: numpy array dos atributos
        :param y: numpy array das classes
        :param batch_size: tamanho do mini-batch (None se for o conjunto completo)
        :param verbose: ativação de uma saída verbosa
        :param validation: tupla contendo o x e y de validação (x_valid, y_valid)
        :return: None
        """

        # Inicializa os pesos das camadas
        self._init_layers(X.shape[1])

        # Variaveis para análise de todas épocas
        costs = []
        accs = []
        costs_val = []
        accs_val = []

        # Para cada época definida realiza uma rodada de treino completo com um batch de dados
        for i in range(epochs):

            # Cria o batch (None se for utilizar o conjunto completo no feedforward)
            x, y_true = self.make_batch(X, y, batch_size)

            # Realiza o feedforward dos dados retornando o output da rede - PREDICT
            y_pred = self.forward(x)

            # Realiza o cálculo de atualização dos pesos - backpropagation
            self.backward(x, y_true, y_pred)

            # Atualiza as camadas da rede
            self.update_layers()

            # Salva os resultados obtidos na época (acurácia e MSE)
            costs.append(self.get_MSELoss(y_pred, y_true))
            accs.append(self.get_accuracy(y_pred, y_true))

            # Calcula as funções de perda e acurácia se tiver um dataset de validação
            if validation is not None:
                x_val, y_val = validation

                # Realiza um feedforward para predição
                y_val_pred = self.forward(x_val)

                # Avalia o resultado obtido pela rede nesta época
                costs_val.append(self.get_MSELoss(y_val_pred, y_val.T))
                accs_val.append(self.get_accuracy(y_val_pred, y_val.T))

            # Printa a saída de cada época se for um fit verboso
            if verbose:
                print("Epoch ", str(i), end=' ')
                print("Loss: ", costs[-1], end=' ')
                print("Acc: ", accs[-1], end=' ')
                if validation is not None:
                    print("Loss Val: ", costs_val[-1], end=' ')
                    print("Acc Val: ", accs_val[-1])
                else:
                    print('')

        return {'loss': costs,
                'accuracy': accs,
                'loss_val': costs_val,
                'accuracy_val': accs_val}

    @staticmethod
    def make_batch(X, y, batch_size):
        """
        Função de criação de um batch de dados aleatorio
        :param X: Conjunto de treino X
        :param y: Conjunto de treino y
        :param batch_size: Tamanho do batch a ser utilizado
        :return:
        """

        # Apenas transpõe o y para uso se for utilzar o batch completo
        if batch_size is None:
            x = X
            y_true = y.T

        # Se não seleciona um batch aleatorio do conjunto de treino (X,y)
        else:
            idx_batch = np.random.randint(0, X.shape[0], batch_size)
            x = X[idx_batch]
            y_true = y[idx_batch].T

        return x, y_true

    def predict(self, X):
        """
        Função de calculo das probabilidades de cada classe para os dados que deseja-se predizer
        :param X: Dados a serem preditos
        :return: Probabilidades dos neuronios de saída
        """
        # Realiza o feedforward com o conjunto x a ser predito
        y_pred = self.forward(X)

        return y_pred.T

    def update_layers(self):
        """
        Atualiza os pesos em todas as camadas
        :return: None
        """

        # Para cada layer atualiza utilizando o gradiente descendente e o learning rate
        for layer in self.layers:
            layer.update_layer(self.learning_rate)

    def forward(self, X):
        """
        Função de feedforward da rede
        :param X: Conjunto de x para predição
        :return: Outputs da última camada da rede
        """

        # Iniciando com o input, para cada camada entra o input
        # atual e retorna o output que é utilizada na próxima camada
        inputs = X.T
        for layer in self.layers:
            inputs = layer.compute_layer(inputs)

        return inputs

    def backward(self, X, y, y_pred):
        """
        Função de backpropagation da rede
        :param X: Conjunto X de entrada no feedforward
        :param y: Conjunto y das classes do conjunto X de entrada
        :param y_pred:Resultado do feedforward para o conjunto X
        :return: None
        """

        # Calcula o erro inicial que será propagado na rede
        dErro = y_pred - y

        # Para cada camada, de trás para frente, realiza o calculo dos gradientes para atualização dos pesos
        for idx in range(len(self.layers) - 1, 0, -1):

            # Define a camada atual
            layer = self.layers[idx]

            # Recupera as ativações da camada anterior
            activations_a = self.layers[idx - 1].activation

            # Realiza os cálculos de backpropagation e salva os resultados para os pesos e os bias
            dErro = layer.backpropagation(dErro, activations_a)

        # Realiza o backpropagation na camada de entrada com os dados de entrada
        layer = self.layers[0]
        _ = layer.backpropagation(dErro, X.T)

    @staticmethod
    def get_accuracy(y_pred, y_true):
        """
        Calcula a acurácia do modelo
        :param y_pred: Y predito pelo modelo
        :param y_true: Y real das intancias
        :return: Acurácia do modelo
        """
        return np.sum(np.equal(np.argmax(y_pred.T, axis=1), np.argmax(y_true.T, axis=1))) / y_true.shape[1]

    @staticmethod
    def get_MSELoss(y_pred, y_true):
        """
        Calculo o erro médio quadrático do modelo
        :param y_pred: Y predito pelo modelo
        :param y_true: Y real das intancias
        :return: Erro médio Quadrático do modelo
        """
        cost = np.mean((y_true - y_pred) ** 2)
        return cost

    def score(self, x, y, verbose=False):
        """
        Calcula o score um conjunto x
        :param x: conjunto de dados x
        :param y: classes do conjunto de dados
        :return:
        """
        y_pred = self.forward(x).T

        y_p = np.argmax(y_pred, axis=-1)
        y_t = np.argmax(y, axis=-1)
        fpr, tpr, thresholds = roc_curve(y_t, y_p, pos_label=2)

        metricas = {
            'accuracy': accuracy_score(y_t, y_p),
            'f1':f1_score(y_t, y_p, average='macro'),
            'balanced accuracy': balanced_accuracy_score(y_t, y_p),
            'auc': auc(fpr, tpr)
        }

        if verbose:
            print('Accuracy', metricas['accuracy'])
            print('F1-Macro', metricas['f1'])
            print('Balanced accuracy', metricas['balanced accuracy'])
            print('AUC', metricas['auc'])

        return metricas

    def __CM(self, y, y_pred):
        return confusion_matrix(y_true=np.argmax(y, axis=-1), y_pred=np.argmax(y_pred, axis=-1))

    def get_confusion_matrix(self, x_treino, x_val, x_teste, y_treino, y_val, y_teste):
        y_pred_treino = self.predict(x_treino)
        y_pred_val = self.predict(x_val)
        y_pred_test = self.predict(x_teste)

        cm_treino = self.__CM(y_treino, y_pred_treino)
        cm_validacao = self.__CM(y_val, y_pred_val)
        cm_teste = self.__CM(y_teste, y_pred_test)

        return cm_treino, cm_validacao, cm_teste