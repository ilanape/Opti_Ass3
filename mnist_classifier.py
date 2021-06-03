import numpy as np
from matplotlib import pyplot as plt

from digit_class import normalization, gradient_descent
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

# normalization
x_train_filtered = normalization(x_train_filtered)
x_test_filtered = normalization(x_test_filtered)

# reshape
n = np.shape(x_train_filtered)[0]
A = np.reshape(x_train_filtered, (n, 784))
A = np.transpose(A)

plt.figure()
plt.xlabel("k iteration")
plt.ylabel("error")
x = np.zeros(784)

plt.title('SD')
x_axis, y_axis = gradient_descent(A, x, y_train_filtered)
plt.semilogy(x_axis, y_axis, label="SD")

# plt.title('Newton')
# x1_axis, y1_axis = newton(A, b, w, y_train)
# plt.semilogy(x1_axis, y1_axis, label="")

plt.legend()
plt.show()
