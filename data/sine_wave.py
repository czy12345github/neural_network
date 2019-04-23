import numpy as np

np.random.seed(2)

T = 20
L = 1000
N = 100

x = np.empty((N, L), 'int64')
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = np.sin(x / 1.0 / T).astype('float64')

a = data[0, :-1]
b = data[0, 1:]

result = open('sin_wave', 'w')
for i in range(len(a)):
    result.write(str(a[i]) + ' ' + str(b[i]) + '\n')

result.close()
