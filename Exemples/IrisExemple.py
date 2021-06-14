from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from N3.Utils import to_categorical
from N3 import Layer, MLP
from N3.Activations import ReLU, Sigmoid


if __name__ == '__main__':
    X = load_iris()['data']
    y = load_iris()['target']

    y_cat = to_categorical(y)
    num_saidas = y_cat.shape[1]
    lr = 0.1

    hidden_layers = [Layer(n_neurons=4, activation=ReLU, initializer='glorotuniform'),
                     Layer(n_neurons=6, activation=ReLU, initializer='glorotuniform'),
                     Layer(num_saidas, Sigmoid, initializer='glorotuniform')]

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.33)

    cls = MLP(layers=hidden_layers, learning_rate=lr)
    cls.fit(X_train, y_train, verbose=False)
    measures = cls.score(X_test, y_test)

    print("Accuracy", measures['accuracy'])
    print("F1-Macro", measures['f1'])
    print("Balanced Accuracy", measures['balanced accuracy'])
    print("AUC", measures['auc'])