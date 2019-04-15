import matplotlib.pyplot as plt
import numpy as np

def main():
    K = 8
    N = 3
    M = 1000
    mean = [[1, 1], [7, 7], [15, 1]]
    cov = [[[12, 0], [0, 1]], [[8, 3], [3, 2]], [[2, 0], [0, 2]]]
    P = [0.6, 0.3, 0.1]
    color = ('ro', 'go', 'bo')

    fig = plt.figure()
    for k in range(K):
        ax = fig.add_subplot(2, K//2, k+1)
        for i in range(N):
            x, y = np.random.multivariate_normal(mean[i], cov[i], int(M * P[i])).T
            ax.plot(x, y, color[i], label='class'+str(i))

        # ax.axis('equal')
        ax.legend(loc=2)
    plt.show()


if __name__=='__main__':
    main()