import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neural_net import NeuralNetwork
from operations import *


def load_dataset(csv_path, target_feature):
    dataset = pd.read_csv(csv_path)
    t = np.expand_dims(
        dataset[target_feature].to_numpy().astype(float), axis=1)
    X = dataset.drop([target_feature], axis=1).to_numpy()
    return X, t


if __name__ == "__main__":
    X, y = load_dataset(
        "/Users/mahongde/Desktop/cs 486/A4/code/data/banknote_authentication.csv", "target")

    np.random.seed(486)

    test_split = 0.2
    X_train = X[:int((1 - test_split) * X.shape[0])]
    X_test = X[int((1 - test_split) * X.shape[0]):]
    y_train = y[:int((1 - test_split) * y.shape[0])]
    y_test = y[int((1 - test_split) * y.shape[0]):]

    X_split_1 = X[:int(test_split * X.shape[0])]
    X_split_2 = X[int(test_split * X.shape[0]):2 *
                  int(test_split * X.shape[0])]
    X_split_3 = X[2 * int(test_split * X.shape[0]):3 *
                  int(test_split * X.shape[0])]
    X_split_4 = X[3 * int(test_split * X.shape[0]):4 *
                  int(test_split * X.shape[0])]
    X_split_5 = X[4 * int(test_split * X.shape[0]):]

    X_train_1 = np.concatenate([X_split_1, X_split_2, X_split_3, X_split_4])
    X_test_1 = np.copy(X_split_5)
    X_train_2 = np.concatenate([X_split_1, X_split_2, X_split_3, X_split_5])
    X_test_2 = np.copy(X_split_4)
    X_train_3 = np.concatenate([X_split_1, X_split_2, X_split_4, X_split_5])
    X_test_3 = np.copy(X_split_3)
    X_train_4 = np.concatenate([X_split_1, X_split_3, X_split_4, X_split_5])
    X_test_4 = np.copy(X_split_2)
    X_train_5 = np.concatenate([X_split_2, X_split_3, X_split_4, X_split_5])
    X_test_5 = np.copy(X_split_1)

    y_split_1 = y[:int(test_split * y.shape[0])]
    y_split_2 = y[int(test_split * y.shape[0]):2 *
                  int(test_split * y.shape[0])]
    y_split_3 = y[2 * int(test_split * y.shape[0]):3 *
                  int(test_split * y.shape[0])]
    y_split_4 = y[3 * int(test_split * y.shape[0]):4 *
                  int(test_split * y.shape[0])]
    y_split_5 = y[4 * int(test_split * y.shape[0]):]

    y_train_1 = np.concatenate([y_split_1, y_split_2, y_split_3, y_split_4])
    y_test_1 = np.copy(y_split_5)
    y_train_2 = np.concatenate([y_split_1, y_split_2, y_split_3, y_split_5])
    y_test_2 = np.copy(y_split_4)
    y_train_3 = np.concatenate([y_split_1, y_split_2, y_split_4, y_split_5])
    y_test_3 = np.copy(y_split_3)
    y_train_4 = np.concatenate([y_split_1, y_split_3, y_split_4, y_split_5])
    y_test_4 = np.copy(y_split_2)
    y_train_5 = np.concatenate([y_split_2, y_split_3, y_split_4, y_split_5])
    y_test_5 = np.copy(y_split_1)

    n_features = X.shape[1]

    net1 = NeuralNetwork(n_features, [64, 64, 32, 32, 16, 1], [
        ReLU(), ReLU(), ReLU(), ReLU(), ReLU(), Sigmoid()], CrossEntropy(), learning_rate=0.01)
    net2 = NeuralNetwork(n_features, [64, 64, 32, 32, 16, 1], [
        ReLU(), ReLU(), ReLU(), ReLU(), ReLU(), Sigmoid()], CrossEntropy(), learning_rate=0.01)
    net3 = NeuralNetwork(n_features, [64, 64, 32, 32, 16, 1], [
        ReLU(), ReLU(), ReLU(), ReLU(), ReLU(), Sigmoid()], CrossEntropy(), learning_rate=0.01)
    net4 = NeuralNetwork(n_features, [64, 64, 32, 32, 16, 1], [
        ReLU(), ReLU(), ReLU(), ReLU(), ReLU(), Sigmoid()], CrossEntropy(), learning_rate=0.01)
    net5 = NeuralNetwork(n_features, [64, 64, 32, 32, 16, 1], [
        ReLU(), ReLU(), ReLU(), ReLU(), ReLU(), Sigmoid()], CrossEntropy(), learning_rate=0.01)
    epochs = 1000

    # net.forward_pass(X_train)
    # a = np.array([-2.55929575e-04, -1.85728500e-04,  7.21304493e-04,
    #               4.74092392e-04,  8.04358627e-04,  9.84004028e-04])
    # s = np.array([-7.67788726e-04, -5.57185499e-04,  2.16391348e-03,
    #              1.42227718e-03,  2.41307588e-03,  2.95201208e-03])

    # print(s / a)
    accuracies = np.zeros((5))

    trained_W_1, epoch_losses_1 = net1.train(X_train_1, y_train_1, epochs)
    ac1 = net1.evaluate(X_test_1, y_test_1, accuracy)
    accuracies[0] = ac1

    trained_W_2, epoch_losses_2 = net2.train(X_train_2, y_train_2, epochs)
    ac2 = net2.evaluate(X_test_2, y_test_2, accuracy)
    accuracies[1] = ac2

    trained_W_3, epoch_losses_3 = net1.train(X_train_3, y_train_3, epochs)
    ac3 = net3.evaluate(X_test_3, y_test_3, accuracy)
    accuracies[2] = ac3

    trained_W_4, epoch_losses_4 = net4.train(X_train_4, y_train_4, epochs)
    ac4 = net4.evaluate(X_test_4, y_test_4, accuracy)
    print("Accuracy on test set 4: {}".format(ac4))
    accuracies[3] = ac4

    trained_W_5, epoch_losses_5 = net5.train(X_train_5, y_train_5, epochs)
    ac5 = net5.evaluate(X_test_5, y_test_5, accuracy)
    accuracies[4] = ac5

    print("accuracy on test set 1: {}".format(ac1))

    print("accuracy on test set 2: {}".format(ac2))

    print("accuracy on test set 3: {}".format(ac3))

    print("accuracy on test set 4: {}".format(ac4))

    print("accuracy on test set 5: {}".format(ac5))

    print("Average of accuracy: {}".format(accuracies.mean()))
    print("standard deviation of accuracy: {}".format(accuracies.std()))

    losses = np.array([epoch_losses_1, epoch_losses_2,
                      epoch_losses_3, epoch_losses_4, epoch_losses_5]).mean(axis=0)
    # print(losses)

    plt.plot(np.arange(0, epochs), losses)

    # plt.plot(np.arange(0, epochs), epoch_losses_1,color='tab:blue',label='fold1')
    # plt.plot(np.arange(0, epochs), epoch_losses_2,color='tab:green',label='fold2')
    # plt.plot(np.arange(0, epochs), epoch_losses_3,color='tab:red',label='fold3')
    # plt.plot(np.arange(0, epochs), epoch_losses_4,color='tab:pink',label='fold4')
    # plt.plot(np.arange(0, epochs), epoch_losses_5,color='tab:orange',label='fold5')
    # plt.legend()
    plt.show()
