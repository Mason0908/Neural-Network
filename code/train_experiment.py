# version 1.1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neural_net import NeuralNetwork
from operations import *

np.random.seed(486)
def load_dataset(csv_path, target_feature):
    dataset = pd.read_csv(csv_path)
    t = np.expand_dims(dataset[target_feature].to_numpy().astype(float), axis=1)
    X = dataset.drop([target_feature], axis=1).to_numpy()
    return X, t

X, y = load_dataset("/Users/mahongde/Desktop/cs 486/A4/code/data/banknote_authentication.csv", "target")

n_features = X.shape[1]
net = NeuralNetwork(n_features, [32,32,16,1], [ReLU(), ReLU(), ReLU(), Sigmoid()], CrossEntropy(), learning_rate=0.01)
epochs = 5000

test_split = 0.2
X_train = X[:int((1 - test_split) * X.shape[0])]
X_test = X[int((1 - test_split) * X.shape[0]):]
y_train = y[:int((1 - test_split) * y.shape[0])]
y_test = y[int((1 - test_split) * y.shape[0]):]

trained_W, epoch_losses = net.train(X_train, y_train, epochs)
print("Accuracy on test set: {}".format(net.evaluate(X_test, y_test, accuracy)))

plt.plot(np.arange(0, epochs), epoch_losses)
plt.show()