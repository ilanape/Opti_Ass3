import numpy as np
from matplotlib import pyplot as plt

from functions import gradient_descent
from loadMNIST_V2 import MnistDataloader

# 4c
mnistDataLoader = MnistDataloader(
    'MNIST\\train-images-idx3-ubyte\\train-images.idx3-ubyte',
    'MNIST\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte',
    'MNIST\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte',
    'MNIST\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte')
(x_train, y_train), (x_test, y_test) = mnistDataLoader.load_data()

x_train = x_train[:30000]
y_train = y_train[:30000]
x_test = x_train[:30000]
y_test = y_train[:30000]

x_train_filtered = []
y_train_filtered = []
x_test_filtered = []
y_test_filtered = []

# 0,1 filter
for i in range(30000):
    if y_train[i] == 0 | y_train[i] == 1:
        x_train_filtered.append(x_train[i])
        y_train_filtered.append(y_train[i])

    if y_test[i] == 0 | y_test[i] == 1:
        x_test_filtered.append(x_test[i])
        y_test_filtered.append(y_test[i])

# 8,9 filter
# for i in range(30000):
#     if y_train[i] == 8 | y_train[i] == 9:
#         x_train_filtered.append(x_train[i])
#         y_train_filtered.append(y_train[i])

#     if y_test[i] == 8 | y_test[i] == 9:
#         x_test_filtered.append(x_test[i])
#         y_test_filtered.append(y_test[i])

# reshape and normalize
A_train = np.reshape(x_train_filtered, (np.shape(x_train_filtered)[0], 784)) / 255
A_train = np.transpose(A_train)
A_test = np.reshape(x_test_filtered, (np.shape(x_test_filtered)[0], 784)) / 255
A_test = np.transpose(A_test)

plt.figure()
plt.xlabel("k iteration")
plt.ylabel("error")
w = np.zeros(784)

plt.title('SD')
x_axis, y_axis = gradient_descent(A_train, w, 1 - y_train_filtered)
x1_axis, y1_axis = gradient_descent(A_test, w, 1 - y_test_filtered)
plt.semilogy(x_axis, y_axis, label="SD train data")
plt.semilogy(x1_axis, y1_axis, label="SD test data")

# # plt.title('Newton')
# # x1_axis, y1_axis = newton(A, b, w, y_train)
# # plt.semilogy(x1_axis, y1_axis, label="")

plt.legend()
plt.show()
