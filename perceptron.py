import numpy as np

def sinal(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0

def perceptron(X, y):
    X = np.array([[1] + x for x in X])
    d = X.shape[1]
    W = np.ones(d).T / d

    l = True
    i = 0
    while l:
        l = False

        for x, y_i in zip(X, y):
            y_classified = sinal(np.dot(x, W))
            if y_classified != y_i:
                W = W + y_i * x
                l = True
        
        i += 1

    print(i)
    return W

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [-1, 1, 1, 1]

W = perceptron(X, y)

for x in X:
    print(x)
    print(sinal(np.dot(x, W[1:]) + W[0]))
