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
# AT = np.transpose(A)
# fn = np.linalg.inv(AT @ A + (80 / 2) * np.transpose(G) @ G) @ AT @ y
# plt.figure()
# plt.plot(x, f)
# plt.plot(x, fn)
# plt.show()

# IRLS
W = np.eye(n)
AT = np.transpose(A)
GT = np.transpose(G)
lamb = 1
eps = 0.001

for i in range(10):
    xi = np.linalg.inv(AT @ A + lamb * GT @ W @ G) @ AT @ y
    W[i:i] = 1 / (np.abs(np.transpose(G[i]) @ xi) + eps)

plt.figure()
plt.plot(x, xi)
plt.show()
