"""
Author:     Rajkumar Conjeevaram Mohan
University: Imperial College London
Email:      rajkumarcm@yahoo.com
Program:    Data compression, and reconstruction using
            Linear Discriminant Analysis ( Simultaneous Diagonalisation )
"""

import numpy as np
import matplotlib.pyplot as plt
from ML.Statistical_ML import pca
from ML.Statistical_ML import data

def lda_sd(X=None,Y=None,retain_threshold=0.1,plot_basic=True,plot_seq=True):

    if X is None:
        X,Y = data.load_data("Iris/iris.data","\n",",",target_col=4,numeric_target=False)

    X = X.T
    classes = np.unique(Y)
    n_classes = len(Y)
    E = []
    size = np.array([0])
    height = size
    width = size
    Nc = np.zeros([n_classes],dtype=np.int)
    for c,i in zip(classes,range(n_classes)):
        Nc[i] = int(np.sum(Y == c))
        temp = np.ones([Nc[i],Nc[i]])/Nc[i]
        E.append(temp)
        size[0] += Nc[i]

    height = height[0]
    width = width[0]
    M = np.zeros([height,width],dtype=np.float)
    si = None
    ei = None
    for c, i in zip(classes, range(n_classes)):
        if i == 0:
            si = 0
            ei = Nc[i]
        else:
            si = ei
            ei = ei + Nc[i]

        M[si:ei,si:ei] = E[i]

    M_N,M_M = M.shape
    I = np.eye(M_N,M_M)
    Xw = np.dot(X,I-M)
    Sw = np.dot(Xw,Xw.T)
    S,U = np.linalg.eig(Sw)

    # Sort the eigen values, and vectors
    indices = np.argsort(np.abs(S))[::-1]
    S = S[indices].real
    U = U[:,indices].real

    # Project the between class vectors ( not the scatter matrix )
    # onto lower dimension
    Xb = np.dot(U.T,np.dot(X,M))
    Q,_,_ = pca.pca(Xb.T,None,retain_threshold=retain_threshold,plot_seq=False,plot_basic=False)
    W = np.dot(U,Q)
    projected_data = np.dot(X.T,W)
    reconstructed_data = np.dot(projected_data,W.T)

    if plot_basic:
        # Plot projected data
        fig, ax = plt.subplots()
        ax.set_xlabel('LD1')
        ax.set_ylabel('LD2')
        ax.set_title("LDA (SD): Projection using %d linear discriminants" % projected_data.shape[1])
        ax.grid()
        plt.tight_layout()
        for label, marker, color in zip(
                classes, ('^', 's', 'o'), ('blue', 'red', 'green')):
            ax.scatter(projected_data[:, 0].real[Y == label],
                       projected_data[:, 1].real[Y == label],
                        c=color, marker=marker, alpha=0.5, label=label)
        ax.legend(loc='upper left', fancybox=True)

        # remove axis spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        plt.show()

if __name__ == '__main__':
    c1 = np.array([[4, 1], [2, 4], [2, 3], [3, 6], [4, 4]]).T
    c2 = np.array([[9, 10], [6, 8], [9, 5], [8, 7], [10, 8]]).T
    X = np.hstack([c1, c2]).T
    Y = np.vstack([np.ones([5, 1]), np.ones([5, 1]) * 2]).reshape([-1])
    lda_sd(X=X, Y=Y, retain_threshold=0, plot_basic=True, plot_seq=False)
    input()