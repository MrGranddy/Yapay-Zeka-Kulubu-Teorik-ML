import matplotlib.pyplot as plt
import numpy as np

# Least Squares Method example data

x = np.random.rand(100)
f = lambda x: 2 * x + 1
y = f(x) + np.random.normal(0, 0.1, 100)

plt.scatter(x, y, s=10, c='b', marker='o')
plt.savefig('lsm_example_data.png')
plt.clf()

# Least Squares Method arsa örneği

x = np.random.rand(100) * 100 + 100
f = lambda x: 100 * x + 1000
y = f(x) + np.random.normal(0, 1000, 100)

plt.scatter(x, y, s=10, c='b', marker='o')
plt.xlabel('Arsa Alanı $m^2$')
plt.ylabel('Arsa Fiyatı $TL$')

plt.savefig('lsm_arsa_ornegi.png')
plt.clf()

# Least Squares Method arsa örneği with line

plt.scatter(x, y, s=10, c='b', marker='o')
plt.xlabel('Arsa Alanı $m^2$')
plt.ylabel('Arsa Fiyatı $TL$')

x = np.linspace(100, 200, 100)
y = 100 * x + 1000
plt.plot(x, y, '-r')

plt.savefig('lsm_arsa_ornegi_with_line.png')
plt.clf()