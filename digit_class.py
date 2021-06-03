import numpy as np
from matplotlib import pyplot as plt

from loadMNIST_V2 import MnistDataloader


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def func(X, w, c1):
    # Linear Regression Objective
    c2 = 1 - c1
    m = np.shape(X)[1]
    sigm_XTw = sigmoid(np.transpose(X) @ w)
    Fw = (-1 / m) * (np.transpose(c1) @ np.log(sigm_XTw)
                     + np.transpose(c2) @ np.log(1 - sigm_XTw))

    # Gradient
    Grad = (1 / m) * X @ (sigm_XTw - c1)

    # Hessian
    D_diag = np.multiply(sigm_XTw, 1 - sigm_XTw)
    D = np.diag(D_diag)
    Hess = (1 / m) * X @ D @ np.transpose(X)

    return Fw, Grad, Hess


def linesearch(x, Fx, grad_x, d, alpha, beta, c, labels, A):
    for j in range(100):
        c1 = np.array(labels)
        Fx_ad, grad_ad, hess_ad = func(A, x + alpha * d, c1, 1 - c1)
        if Fx_ad <= Fx + c * alpha * np.dot(grad_x, d):
            return alpha
        else:
            alpha = beta * alpha


def gradient_descent(A, x, labels):
    c1 = np.array(labels)
    Fx1, grad_x1, hessx1 = func(A, x, c1, 1 - c1)
    initial_grad_norm = np.linalg.norm(grad_x1)

    x_axis = []
    y_axis = []
    for i in range(100):
        print(i)
        # define weight
        c1 = np.array(labels)
        Fx, grad_x, hess_x = func(A, x, c1, 1 - c1)
        d = -grad_x
        alpha = linesearch(x, Fx, grad_x, grad_x, 0.25, 0.5, 1e-4, labels, A)

        x_axis.append(i)
        y_axis.append(np.abs(Fx))

        # apply iteration
        x = x + alpha * d
        x = np.clip(x, -1, 1)

        # Convergence criterion
        if (np.linalg.norm(grad_x) / initial_grad_norm) < 0.1:
            break

    return x_axis, y_axis


def newton(A, b, x, labels):
    c1 = np.array(labels)
    Fx1, grad_x1, hessx1 = func(A, x, c1, 1 - c1)
    initial_grad_norm = np.linalg.norm(grad_x1)

    x_axis = []
    y_axis = []

    for i in range(100):
        # define weight
        Fx, grad_x, hess_x = func(A, x, c1, 1 - c1)
        d = -np.linalg.inv(hess_x) @ grad_x
        alpha = linesearch(x, Fx, grad_x, d, 1, 0.5, 1e-4)

        x_axis.append(i)
        y_axis.append(np.abs(Fx))

        # apply iteration
        x = x + alpha * d
        np.clip(x, -1, 1)

        # Convergence criterion
        if (np.linalg.norm(grad_x) / initial_grad_norm) < 0.1:
            break

    return x_axis, y_axis


def normalization(vectors):
    normalized_vectors = []
    for i in range(len(vectors)):
        normalized_vectors.append(np.array(vectors[i]).flatten())
        for j in range(len(vectors[i])):
            normalized_vectors[i][j] /= 255
    return np.array(normalized_vectors)

