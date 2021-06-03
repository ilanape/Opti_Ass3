import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def func(X, w, labels):
    # Linear Regression Objective
    c1 = labels
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


def linesearch(x, Fx, grad_x, d, alpha, beta, c, c1, A):
    for j in range(100):
        Fx_ad, grad_ad, hess_ad = func(A, x + alpha * d, c1)
        if Fx_ad <= Fx + c * alpha * np.dot(grad_x, d):
            return alpha
        else:
            alpha = beta * alpha


def gradient_descent(A, x, labels):
    c1 = np.array(labels)
    x_axis = []
    y_axis = []
    for i in range(100):
        print(i)
        # define weight
        Fx, grad_x, hess_x = func(A, x, c1)
        d = -grad_x
        alpha = linesearch(x, Fx, grad_x, grad_x, 0.25, 0.5, 1e-4, c1, A)

        x_axis.append(i)
        y_axis.append(np.abs(Fx))

        # apply iteration
        x = x + alpha * d
        x = np.clip(x, -1, 1)

        # Convergence criterion
        if Fx < 0.001:
            break

    return x_axis, y_axis


def newton(A, b, x, labels):
    c1 = np.array(labels)
    x_axis = []
    y_axis = []

    for i in range(100):
        # define weight
        Fx, grad_x, hess_x = func(A, x, c1)
        d = -np.linalg.inv(hess_x) @ grad_x
        alpha = linesearch(x, Fx, grad_x, d, 1, 0.5, 1e-4)

        x_axis.append(i)
        y_axis.append(np.abs(Fx))

        # apply iteration
        x = x + alpha * d
        np.clip(x, -1, 1)

        # Convergence criterion
        if Fx < 0.001:
            break

    return x_axis, y_axis

