import numpy as np
from scipy.sparse import spdiags
import matplotlib.pyplot as plt

x = np.arange(0, 5, 0.01)
n = np.size(x)
one = int(n / 5)
f = np.zeros(x.shape)
f[0:one] = 0.0 + 0.5 * x[0:one]
f[one:2 * one] = 0.8 - 0.2 * np.log(x[100:200])
f[(2 * one):3 * one] = 0.7 - 0.2 * x[(2 * one):3 * one]
f[(3 * one):4 * one] = 0.3
f[(4 * one):(5 * one)] = 0.5 - 0.1 * x[(4 * one):(5 * one)]
G = spdiags([-np.ones(n), np.ones(n)], np.array([0, 1]), n, n).toarray()
A = np.eye(n)
etta = 0.1 * np.random.randn(np.size(x))
y = f + etta

# original
# plt.figure()
# plt.plot(x, y)
# plt.plot(x, f)
# plt.show()

# regularized non-weighted LS L2
AT = np.transpose(A)
lamb = 80
fn = np.linalg.inv(AT @ A + (lamb / 2) * np.transpose(G) @ G) @ AT @ y
plt.figure()
# plt.plot(x, f, 'b', label="Original")
plt.plot(x, fn, 'g',  label="LS")
# plt.show()

# IRLS
W = np.eye(n)
GT = np.transpose(G)
lamb = 1
eps = 0.001

for k in range(10):
    x_k = np.linalg.inv(AT @ A + (lamb / 2) * GT @ W @ G) @ AT @ y
    Gx_k = G @ x_k
    W_diag = [1 / (np.abs(Gx_k[i]) + eps) for i in range(n)]
    W = np.diag(W_diag)

# plt.figure()
plt.plot(x, x_k, 'r', label="IRLS")
plt.legend()
plt.show()
