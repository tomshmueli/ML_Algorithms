import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
import ssl
import numpy.random

ssl._create_default_https_context = ssl._create_unverified_context

mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']

idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]


# question a
def kNN_Alg(train_images, train_labels, image, k):
    distances = np.array([np.linalg.norm(train_image - image) for train_image in train_images])
    indexesOfKSmallest = np.argpartition(distances, k)[:k]
    return str(np.argmax(np.bincount(train_labels[indexesOfKSmallest].astype(int))))


# question b
def accuracy(n, k):
    cur_train = train[:n]
    cur_train_labels = train_labels[:n]
    correctness_of_guesses = np.array(
        [kNN_Alg(cur_train, cur_train_labels, image, k) == test_labels[i] for i, image in enumerate(test)])
    return np.count_nonzero(correctness_of_guesses) / correctness_of_guesses.size


n = 1000
k = 10
print("Question b: The accuracy of the prediction is:", accuracy(n, k))

# question c:
accuracies = np.array([accuracy(1000, k) for k in range(1, 101)])
ks = list(range(1, 101))
plt.plot(ks, accuracies)
plt.xlabel("k")
plt.ylabel("accuracy")
plt.title('Prediction accuracy as a function of k')
plt.show()

# question d:
accuracies = np.array([accuracy(n, 1) for n in range(100, 5001, 100)])
ns = list(range(100, 5001, 100))
plt.plot(ns, accuracies)
plt.xlabel("n")
plt.ylabel("accuracy")
plt.title('Prediction accuracy as a function of n')
plt.show()
