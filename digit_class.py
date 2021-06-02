import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def func(X, w, c1, c2):
    # Linear Regression Objective
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

def linesearch(x, Fx, grad_x, d, alpha, beta, c):
    for j in range(100):
        c1 =
        Fx_ad, grad_ad, hess_ad = func(A, x+alpha*d, c1, 1-c1)
        if Fx_ad <= Fx + c*alpha*np.dot(grad_x, d):
            return alpha
        else:
            alpha = beta * alpha


def gradient_descent(A, b, x):
    c1 =
    Fx1, grad_x1, hessx1 = func(A, x, c1, 1 - c1)
    initial_grad_norm = np.linalg.norm(grad_x1)
    cb =
    Fb, grad_b, hess_b = func(A, b, cb, 1 - cb)

    x_axis = []
    y_axis = []
    for i in range(100):
        # define weight
        c1 =
        Fx, grad_x, hess_x = func(A, x, c1, 1 - c1)
        d = -grad_x
        alpha = linesearch(x, Fx, grad_x ,grad_x, 0.25, 0.5, 1e-4)

        x_axis.append(i)
        y_axis.append(np.abs(Fx - Fb))

        # apply iteration
        x = x + alpha * d
        np.clip(x, -1, 1)

        # Convergence criterion
        if (np.linalg.norm(grad_x) / initial_grad_norm) < 0.1:
            break

    return x_axis, y_axis



def newton(A, b, x):
    c1 =
    Fx1, grad_x1, hessx1 = func(A, x, c1, 1 - c1)
    initial_grad_norm = np.linalg.norm(grad_x1)
    cb =
    Fb, grad_b, hess_b = func(A, b, cb, 1 - cb)

    x_axis = []
    y_axis = []

    for i in range(100):
        # define weight
        c1 =
        Fx, grad_x, hess_x = func(A, x, c1, 1 - c1)
        d = -np.linalg.inv(hess_x) @ grad_x
        alpha = linesearch(x, Fx, grad_x, d, 1, 0.5, 1e-4)

        x_axis.append(i)
        y_axis.append(np.abs(Fx - Fb))

        # apply iteration
        x = x + alpha * d
        np.clip(x, -1, 1)

        # Convergence criterion
        if (np.linalg.norm(grad_x) / initial_grad_norm) < 0.1:
            break

    return x_axis, y_axis


# plots

# 4b
# n = 20
# m = 20
# w = np.random.rand(n)
# d = np.random.rand(n)
# epsilon = 0.1
# X = np.random.rand(n, m)
# c1 = np.random.randint(2, size=m)
# c2 = 1 - c1
#
# x_axis = []
# y_axis = []
# y1_axis = []
# y2_axis = []
# y3_axis = []
#
# for k in range(1, 9):
#     eps = epsilon * (0.5 ** k)
#     x_axis.append(k)
#     Fx, gradx, hessx = func(X, w, c1, c2)
#     Fx_ed, gradx_ed, hessx_ed = func(X, w + eps * d, c1, c2)
#
#     # Gradient test
#     y_axis.append(np.abs(Fx_ed - Fx))
#     y1_axis.append(np.abs(Fx_ed - Fx - eps * np.transpose(d) @ gradx))
#
#     # Jacobian test
#     y2_axis.append(np.linalg.norm(gradx_ed - gradx))
#     y3_axis.append(np.linalg.norm(gradx_ed - gradx - eps * hessx @ d))
#
# plt.figure()
#
# plt.xlabel("k iteration")
# plt.ylabel("error")

# plt.title('Gradient verification test')
# plt.semilogy(x_axis, y_axis, 'r', label="|f(x+ed)-f(x)|")
# plt.semilogy(x_axis, y1_axis, 'g', label="|f(x+ed)-f(x) - e*d*grad(x)|")

# plt.title('Jacobian verification test')
# plt.semilogy(x_axis, y2_axis, 'b', label="||grad(x+ed)-grad(x)||")
# plt.semilogy(x_axis, y3_axis, 'orange', label="||grad(x+ed)-grad(x) - e*hess(x)*d||")
#
# plt.legend()
# plt.show()

# 4c
plt.figure()
plt.xlabel("k iteration")
plt.ylabel("error")

plt.title('SD')
x_axis, y_axis = gradient_descent(A, w, b)
plt.semilogy(x_axis, y_axis, label="")

plt.title('Newton')
x1_axis, y1_axis = newton(A, w, b)
plt.semilogy(x1_axis, y1_axis, label="")

plt.legend()
plt.show()
