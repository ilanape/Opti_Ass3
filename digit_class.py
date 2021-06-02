import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def func(X, w, c1, c2):
    # Linear Regression Objective
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


# Gradient test
n = 20
m = 20
w = np.random.rand(n)
d = np.random.rand(n)
epsilon = 0.1
X = np.random.rand(n, m)
c1 = np.random.randint(2, size=m)
c2 = 1 - c1

x_axis = []
y_axis = []
y1_axis = []
y2_axis = []
y3_axis = []

for k in range(1, 9):
    eps = epsilon * (0.5 ** k)
    x_axis.append(k)
    Fx, gradx, hessx = func(X, w, c1, c2)
    Fx_ed, gradx_ed, hessx_ed = func(X, w + eps * d, c1, c2)

    y_axis.append(np.abs(Fx_ed - Fx))
    y1_axis.append(np.abs(Fx_ed - Fx - eps * np.transpose(d) @ gradx))
    y2_axis.append(np.linalg.norm(gradx_ed - gradx))
    y3_axis.append(np.linalg.norm(gradx_ed - gradx - eps * hessx @ d))

plt.figure()
plt.title('Gradient and Jacobian verification tests')
plt.xlabel("k iteration")
plt.ylabel("error")
plt.semilogy(x_axis, y_axis, 'r', label="|f(x+ed)-f(x)|")
plt.semilogy(x_axis, y1_axis, 'g', label="|f(x+ed)-f(x) - e*d*grad(x)|")
plt.semilogy(x_axis, y2_axis, 'b', label="||grad(x+ed)-grad(x)||")
plt.semilogy(x_axis, y3_axis, 'orange', label="||grad(x+ed)-grad(x) - e*hess(x)*d||")
plt.legend()
plt.show()
