import numpy as np
import matplotlib.pyplot as plt

def sinal(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0

def perceptron(X, y):
    X = np.array([[1] + x for x in X]) # adicionando 1 na frente de tudo para que o peso correspondente seja o bias
    d = X.shape[1]
    W = np.ones(d).T

    l = True
    while l:
        l = False

        for x, y_i in zip(X, y):
            y_classified = sinal(np.dot(x, W))
            if y_classified != y_i:
                W = W + y_i * x
                l = True

    return W

def plot_decision_boundary(X, y, W, title):
    # Simplified: just plot the points and the hyperplane line for W
    plt.figure()
    plt.title(title)
    # Plot filled for y==1, hollow for y==-1
    X_1 = [x for x, label in zip(X, y) if label == 1]
    X_m1 = [x for x, label in zip(X, y) if label == -1]
    plt.scatter([x[0] for x in X_1], [x[1] for x in X_1], c='k', s=60, edgecolors='w', label='y=1', marker='o')
    plt.scatter([x[0] for x in X_m1], [x[1] for x in X_m1], facecolors='none', edgecolors='k', s=60, label='y=-1', marker='o')
    # Add perceptron output as label for each point
    for x in X:
        y_pred = sinal(np.dot([1] + x, W))
        plt.text(x[0]+0.05, x[1]+0.05, str(y_pred), fontsize=12, color='blue')
    x_min, x_max = -0.5, 1.5
    xs = np.linspace(x_min, x_max, 100)
    if abs(W[2]) > 1e-8:
        ys = -(W[0] + W[1]*xs) / W[2]
        plt.plot(xs, ys, color='r', label='Decision boundary')
    plt.xlim(x_min, x_max)
    plt.ylim(x_min, x_max)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_or = [-1, 1, 1, 1]
    y_and = [-1, -1, -1, 1]
    y_nand = [1, 1, 1, -1]

    W_or = perceptron(X, y_or)
    W_and = perceptron(X, y_and)
    W_nand = perceptron(X, y_nand)

    plot_decision_boundary(X, y_or, W_or, 'OR Gate')
    plot_decision_boundary(X, y_and, W_and, 'AND Gate')
    plot_decision_boundary(X, y_nand, W_nand, 'NAND Gate')
