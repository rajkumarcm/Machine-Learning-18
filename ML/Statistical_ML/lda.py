"""
Author: Rajkumar Conjeevaram Mohan
Email: rajkumarcm@yahoo.com
Program: Data compression, and reconstruction using
         Linear Discriminant Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from ML.Statistical_ML import data

def lda(A=None,Y=None,retain_threshold=0.1,plot_basic=True,plot_seq=True):

    if A is None:
        A, Y = data.load_data("Iris/iris.data", "\n", ",", target_col=4, numeric_target=False)
        # A,Y = data.load_data("breast_cancer_kaggle.csv","\n",",",1)
        #
        # # DELETE THIS WHEN DIFFERENT DATASET IS USED----------------
        # A = A[:,1:]
        # #-----------------------------------------------------------

    N,M = A.shape
    classes = np.unique(Y)
    means = np.zeros([len(classes),M])

    for i,c in zip(range(len(classes)),classes):
        indices = np.reshape(np.argwhere(Y == c),[-1])
        means[i] = np.mean(A[indices],axis=0)

    """ Between Class Scatter Matrix
    If the number of classes is two, then change the way
    S_B i.e., the between class scatter matrix is computed
    - to  (mean(c1) - mean(c2))^2
    """
    S_B = np.zeros([M,M])
    mean = np.mean(means,axis=0)
    if len(classes) != 2:
        for i,c in zip(range(len(classes)),classes):
            n = A[Y==c].shape[0]
            temp = np.reshape(means[i]-mean,[1,-1])
            S_B += n * np.dot(temp.T,temp)
    else:
        temp = means[0]-means[1]
        S_B += np.dot(temp.T,temp)

    """
    Within Class Scatter Matrix
    """
    S_W = np.zeros([M,M])
    for i,c in zip(range(len(classes)),classes):
        indices = np.reshape(np.argwhere(Y == c),[-1])
        temp = A[indices] - means[i]
        S_W += (np.dot(temp.T,temp))/len(indices)

    """
    Eigen-Analysis
    """
    temp = np.dot(np.linalg.inv(S_W),S_B)
    S,V = np.linalg.eig(temp)
    S = np.real(S)
    V = np.real(V)
    indices = np.argsort(np.abs(S))[::-1]
    S = S[indices]
    V = V[:,indices]
    Z = A.shape[0] * A.shape[1]
    variance = np.zeros([M])
    variance[0] = np.Inf
    variance[1] = np.Inf

    S_total = np.sum(S)
    S_strength = np.array([s/S_total for s in S],dtype=np.float32)
    S_indices = (S_strength >= retain_threshold)
    V_new = V[:,S_indices]

    projected_data = np.dot(A,V_new)
    reconstructed_data = np.dot(projected_data,V_new.T)

    if plot_basic:
        # Plot projected data
        fig, ax = plt.subplots()
        plt.xlabel('LD1')
        plt.ylabel('LD2')
        plt.title("LDA: Projection using %d linear discriminants" % V_new.shape[1])
        plt.grid()
        plt.tight_layout()
        for label, marker, color in zip(
                classes, ('^', 's', 'o'), ('blue', 'red', 'green')):
            plt.scatter(projected_data[:, 0].real[Y == label],
                        projected_data[:, 1].real[Y == label],
                        c=color, marker=marker, alpha=0.5, label=label)
        plt.legend(loc='upper left', fancybox=True)

        # remove axis spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    """----------------------------------------
    Original Data plotting
    ----------------------------------------"""
    fig, ax = plt.subplots()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title("Original Data")
    plt.grid()
    plt.tight_layout()
    for label, marker, color in zip(
            classes, ('^', 's', 'o'), ('blue', 'red', 'green')):
        plt.scatter(A[:, 0].real[Y == label],
                    A[:, 1].real[Y == label],
                    c=color, marker=marker, alpha=0.5, label=label)
    plt.legend(loc='upper left', fancybox=True)

    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


    """----------------------------------------
    Only run the following code, when you want
    to see the sequential plot of projection
    onto latent space
    ----------------------------------------"""
    if plot_seq:
        for i in range(2,M+1):
            temp = V_new[:,:i]
            projected = np.dot(A,temp)
            # reconstructed = np.dot(projected,V.T[:i,:])
            reconstructed = np.dot(projected, temp.T)
            variance[i-1] = np.sum((A - reconstructed)**2)/Z

            # Plot projected data
            fig,ax = plt.subplots()
            plt.xlabel('LD1')
            plt.ylabel('LD2')
            plt.title("LDA: Projection using %d discriminants"%(i))
            plt.grid()
            plt.tight_layout()
            for label, marker, color in zip(
                    classes, ('^', 's', 'o'), ('blue', 'red', 'green')):
                plt.scatter(projected_data[:, 0].real[Y == label],
                            projected_data[:, 1].real[Y == label],
                            c=color, marker=marker, alpha=0.5, label=label)
            plt.legend(loc='upper left', fancybox=True)

            # remove axis spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)

        fig,ax = plt.subplots()
        plt.xlabel("Component")
        plt.ylabel("Variance")
        plt.grid()
        plt.tight_layout()
        plt.title("LDA Reconstruction error")
        plt.scatter(range(1,M+1),variance,c='r',linewidths=0.5)

        # remove axis spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    if plot_basic or plot_seq:
        plt.show()

    return V_new,projected_data,reconstructed_data

if __name__ == '__main__':
    _,_,_ = lda(retain_threshold=0.008,plot_basic=False,plot_seq=True)
    input()