#################################
# Your name:Tom Shmueli - 315363473
#################################
import math

import numpy as np
import numpy.random
from matplotlib import pyplot as plt
from scipy.special import softmax
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    w = numpy.zeros(data.shape[1])  # Initialize weights
    for t in range(1, T + 1):
        i = numpy.random.randint(len(data))  # Sample uniformly
        y = labels[i]
        x = data[i]
        eta = eta_0 / t
        if y * numpy.dot(w, x) < 1:
            # Update weights if misclassified
            w = (1 - eta) * w + (eta * C * y * x)
        else:
            # Otherwise, update weights without the hinge loss term
            w = (1 - eta) * w
    return w


def SGD_log(data, labels, eta_0, T):
    w = np.zeros(data.shape[1])  # Initialize weights
    norms = []  # List to store norms of w
    for t in range(1, T + 1):
        i = np.random.randint(0, len(data))  # Sample uniformly
        x = data[i]
        y = labels[i]
        eta = eta_0 / t  # Calculate the learning rate for this iteration
        y_x_w = -y * np.dot(x, w)  # Calculate margin

        sigmoid_value = stable_sigmoid(y_x_w)  # Compute sigmoid for gradient
        gradient = -y * x * sigmoid_value  # Calculate gradient
        w = w - eta * gradient  # Update weights

        norm_w = np.linalg.norm(w)  # Compute the norm of w
        norms.append(norm_w)  # Store the norm of w for this iteration
    return w, norms


#################################

# Place for additional code

#################################

def hinge_loss(w, x, y, C):
    # Calculate hinge loss with L2 regularization
    loss = C * max(0, 1 - y * numpy.dot(w, x)) + 0.5 * numpy.linalg.norm(w) ** 2
    return loss


def stable_sigmoid(y_x_w):
    if y_x_w >= 0:
        z = np.exp(-y_x_w)
        return 1 / (1 + z)
    else:
        z = np.exp(y_x_w)
        return z / (1 + z)


def evaluate_accuracy(w, data, labels):
    predictions = np.sign(np.dot(data, w))
    correct_predictions = np.sum(predictions == labels)
    accuracy = correct_predictions / len(labels)
    return accuracy


def q1_a_best_eta(C=1, T=1000, iterations=10):
    # when zooming in on result use:
    # eta_values = [0.1 * i for i in range(1, 20)]

    # Generate eta values from 10^-5 to 10^5
    eta_values = np.logspace(-5, 5, num=11)
    accuracies = []

    for eta in eta_values:
        sum_accuracy = 0
        for i in range(iterations):
            w = SGD_hinge(train_data, train_labels, C, eta, T)
            sum_accuracy += evaluate_accuracy(w, validation_data, validation_labels)
        accuracies.append(sum_accuracy / iterations)

    # Find the best eta and its corresponding accuracy
    opt_eta_index = np.argmax(accuracies)
    opt_eta = eta_values[opt_eta_index]
    best_accuracy = accuracies[opt_eta_index]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.title("Accuracy of SGD for Hinge loss as a function of eta_0")
    plt.xlabel('eta_0')
    plt.ylabel('Average accuracy')
    plt.xscale('log')
    plt.plot(eta_values, accuracies, marker='o')

    # Annotating below the x-axis
    # Annotating within the plot area at a fixed position towards the bottom
    plt.text(0.5, 0.1, f'Best eta: {opt_eta:.4f}\nAccuracy: {best_accuracy:.4f}',
             horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes,  # This ensures the positioning is relative to the axes
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

    plt.show()
    return opt_eta  # return best eta --> maximal accuracy


def q1_b_best_c(eta, T=1000, iterations=10):
    # Generate c values from 10^-5 to 10^5
    c_values = np.logspace(-5, 5, num=11)
    accuracies = []

    for c in c_values:
        sum_accuracy = 0
        for i in range(iterations):
            w = SGD_hinge(train_data, train_labels, c, eta, T)
            sum_accuracy += evaluate_accuracy(w, validation_data, validation_labels)
        accuracies.append(sum_accuracy / iterations)

    # Find the best eta and its corresponding accuracy
    opt_c_index = np.argmax(accuracies)
    opt_c = c_values[opt_c_index]
    best_accuracy = accuracies[opt_c_index]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.title("Accuracy of SGD for Hinge loss as a function of C")
    plt.xlabel('C')
    plt.ylabel('Average accuracy')
    plt.xscale('log')
    plt.plot(c_values, accuracies, marker='o')

    # Annotating below the x-axis
    # Annotating within the plot area at a fixed position towards the bottom
    plt.text(0.5, 0.1, f'Best C: {opt_c:.4f}\nAccuracy: {best_accuracy:.4f}',
             horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes,  # This ensures the positioning is relative to the axes
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

    plt.show()

    return opt_c


def q1_c_w_img(eta, c, T=20000):
    w = SGD_hinge(train_data, train_labels, c, eta, T)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.show()
    return w


def q1_d_accuracy_on_test_set(w):
    return evaluate_accuracy(w, test_data, test_labels)


def q2_a_best_eta(T=1000, iterations=10):
    # Generate eta values from 10^-5 to 10^5
    eta_values = np.logspace(-6, 6, num=20)
    accuracies = []

    for eta in eta_values:
        sum_accuracy = 0
        for i in range(iterations):
            w, norms = SGD_log(train_data, train_labels, eta, T)
            sum_accuracy += evaluate_accuracy(w, validation_data, validation_labels)
        accuracies.append(sum_accuracy / iterations)

    # Find the best eta and its corresponding accuracy
    opt_eta_index = np.argmax(accuracies)
    opt_eta = eta_values[opt_eta_index]
    best_accuracy = accuracies[opt_eta_index]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.title("Accuracy of SGD for Hinge loss as a function of eta_0")
    plt.xlabel('eta_0')
    plt.ylabel('Average accuracy')
    plt.xscale('log')
    plt.plot(eta_values, accuracies, marker='o')

    # Annotating below the x-axis
    # Annotating within the plot area at a fixed position towards the bottom
    plt.text(0.5, 0.1, f'Best eta: {opt_eta:.7f}\nAccuracy: {best_accuracy:.4f}',
             horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes,  # This ensures the positioning is relative to the axes
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

    plt.show()
    return opt_eta  # return best eta --> maximal accuracy


def q2_b_w_img(eta, T=20000):
    w, norms = SGD_log(train_data, train_labels, eta, T)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.show()
    return w, norms


def q2_c_norms(norms):
    plt.figure(figsize=(10, 6))
    plt.plot(norms, label='Norm of w')
    plt.title('Norm of w over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Norm of w')
    plt.legend()
    plt.grid(True)
    plt.show()
    return


train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
# QUESTION 1
best_eta = q1_a_best_eta()
best_c = q1_b_best_c(best_eta)
best_w = q1_c_w_img(best_eta, best_c)
accuracy_best_w = q1_d_accuracy_on_test_set(best_w)
print(f"The accuracy on the test set: {accuracy_best_w}")

# # QUESTION 2
best_eta_2 = q2_a_best_eta()
w_2, norms_2 = q2_b_w_img(best_eta_2)
q2_c_norms(norms_2)
