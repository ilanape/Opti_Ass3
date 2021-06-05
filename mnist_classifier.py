import numpy as np
from matplotlib import pyplot as plt

from functions import gradient_descent, newton
from loadMNIST_V2 import MnistDataloader

# 4c

mnistDataLoader = MnistDataloader(
    'MNIST\\train-images-idx3-ubyte\\train-images.idx3-ubyte',
    'MNIST\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte',
    'MNIST\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte',
    'MNIST\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte')
(x_train, y_train), (x_test, y_test) = mnistDataLoader.load_data()

x_train_filtered = []
y_train_filtered = []
x_test_filtered = []
y_test_filtered = []

# 0,1 filter
for i in range(30000):
    if y_train[i] == 0 or y_train[i] == 1:
        x_train_filtered.append(x_train[i])
        y_train_filtered.append(y_train[i])

for i in range(len(y_test)):
    if y_test[i] == 0 or y_test[i] == 1:
        x_test_filtered.append(x_test[i])
        y_test_filtered.append(y_test[i])

# 8,9 filter
for i in range(30000):
    if y_train[i] == 8 or y_train[i] == 9:
        x_train_filtered.append(x_train[i])
        y_train_filtered.append(y_train[i])

for i in range(len(y_test)):
    if y_test[i] == 8 or y_test[i] == 9:
        x_test_filtered.append(x_test[i])
        y_test_filtered.append(y_test[i])

# normalize and to array
x_train_filtered = np.asarray(x_train_filtered) / 255
x_test_filtered = np.asarray(x_test_filtered) / 255
y_train_filtered = np.asarray(y_train_filtered)
y_test_filtered = np.asarray(y_test_filtered)

# reshape
num = np.shape(x_train_filtered)[0]
A_train = np.empty((num, 784))
for i in range(num):
    A_train[i] = x_train_filtered[i].flatten()

num1 = np.shape(x_test_filtered)[0]
A_test = np.empty((num1, 784))
for i in range(num1):
    A_test[i] = x_test_filtered[i].flatten()

# images in columns
A_train = np.transpose(A_train)
A_test = np.transpose(A_test)
w = np.zeros(784)

plt.figure()
plt.xlabel("k iteration")
plt.ylabel("|F(w_k) - F(w*)|")

# Gradient
# plt.title('Gradient Descent 0 1')

# plt.title('Gradient Descent Objective 8 9')
# y_train_filtered = y_train_filtered % 8
# y_test_filtered = y_test_filtered % 8
# #
# x_axis, y_axis = gradient_descent(A_train, w, y_train_filtered)
# x1_axis, y1_axis = gradient_descent(A_test, w, y_test_filtered)
# plt.semilogy(x_axis, np.abs(y_axis - y_axis[len(y_axis)-1]), 'green', label='Train data')
# plt.semilogy(x_axis, np.abs(y1_axis - np.min(y1_axis)), 'red', label='Test data')

# Newton
plt.title('Exact Newton 0 1')

# plt.title('Exact Newton 8 9')
# y_train_filtered = y_train_filtered % 8
# y_test_filtered = y_test_filtered % 8
# #
x_axis, y_axis = newton(A_train, w, y_train_filtered)
x1_axis, y1_axis = newton(A_test, w, y_test_filtered)
plt.semilogy(x_axis, np.abs(y_axis - y_axis[len(y_axis)-1]), 'green', label='Train data')
plt.semilogy(x1_axis, np.abs(y1_axis - np.min(y1_axis)), 'red', label='Test data')

plt.legend()
plt.show()
