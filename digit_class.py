import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def func(X, w, c1, c2):
    m = np.shape(X)[1]
    sigm_XTw = [sigmoid(np.traspose(X[:, i]) @ w) for i in range(m)]
    Fw = (-1 / m) * (np.transpose(c1) * np.log(sigm_XTw)
                     + np.transpose(c2) * np.log(1 - sigm_XTw))

    Grad = X @ (sigm_XTw - c1)

    D_diag = [sigm_XTw[i] * (1-sigm_XTw[i]) for i in range(m)]
    D = np.diag(D_diag)
    Hess = X @ D @ np.transpose(X)

    return Fw, Grad, Hess
