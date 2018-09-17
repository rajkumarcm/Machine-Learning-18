"""
Author: Rajkumar Conjeevaram Mohan
Email: rajkumarcm@yahoo.com
Program: Data compression, and reconstruction using
         Principal Component Analysis
"""

import numpy as np
from ML.Statistical_ML import data
import matplotlib.pyplot as plt

def pca(X=None,Y=None,image=False,retain_threshold=0.1,plot_basic=False,plot_seq=True):

    if X is None:
        X,Y = data.load_data("Iris/iris.data","\n",",",4)

    N,M = X.shape
    mean = np.reshape(np.mean(X,axis=0),[1,-1])
    X -= mean
    cov = np.dot(X.T,X)/N
    S,V = np.linalg.eig(cov)
    S = np.real(S)
    V = np.real(V)
    indices = np.argsort(np.abs(S))[::-1]
    S = S[indices]
    V = V[:,indices]
    S_total = np.sum(S)
    S_strength = np.array([s / S_total for s in S], dtype=np.float32)
    S_indices = (S_strength >= retain_threshold)
    V_new = V[:, S_indices]

    n_comp = V.shape[1]
    variance = np.zeros([n_comp])
    variance[0] = np.Inf
    variance[1] = np.Inf

    projected_data = np.dot(X, V_new)
    reconstructed_data = np.dot(projected_data, V_new.T)

    if plot_basic:
        # Plot projected data
        fig, ax = plt.subplots()
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title("PCA: Projection using %d components" %V_new.shape[1])
        plt.grid()
        plt.tight_layout()
        for label, marker, color in zip(
                np.unique(Y), ('^', 's', 'o'), ('blue', 'red', 'green')):
            plt.scatter(projected_data[:, 0].real[Y == label],
                        projected_data[:, 1].real[Y == label],
                        c=color, marker=marker, alpha=0.5, label=label)
        plt.legend(loc='upper left', fancybox=True)

        # remove axis spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    if plot_seq:
        for i in range(2,n_comp):
            projected = np.dot(X,V[:,:i])
            reconstructed = np.dot(projected,V[:,:i].T)
            temp = (X - reconstructed)**2
            variance[i] = np.sum(temp)/(N*M)

            # Plot projected data
            fig, ax = plt.subplots()
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title("PCA: Projection using %d components" % (i + 1))
            plt.grid()
            plt.tight_layout()
            for label, marker, color in zip(
                    np.unique(Y), ('^', 's', 'o'), ('blue', 'red', 'green')):
                plt.scatter(projected[:, 0].real[Y == label],
                            projected[:, 1].real[Y == label],
                            c=color, marker=marker, alpha=0.5, label=label)
            plt.legend(loc='upper left', fancybox=True)

            # remove axis spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)

        if not image:
            # Plot the Original Data
            plt.figure()
            plt.title("Original Data")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.grid()
            plt.tight_layout()
            for label, marker, color in zip(
                    np.unique(Y), ('^', 's', 'o'), ('blue', 'red', 'green')):
                plt.scatter(X[:, 0].real[Y == label],
                            X[:, 1].real[Y == label],
                            c=color, marker=marker, alpha=0.5)

            # Plot the reconstruction error
            plt.figure()
            plt.title("Reconstruction error of PCA")
            plt.xlabel("Principal Component")
            plt.ylabel("Variance")
            plt.grid()
            plt.tight_layout()
            plt.plot(range(1, n_comp + 1), variance, '-or', linewidth=1)
            plt.show()
        else:
            plt.figure()
            plt.title("Original Image")
            plt.imshow(X)

            plt.figure()
            plt.title("Image Compressed using PCA")
            plt.imshow(reconstructed_data)





    #Plotting-----------------------------------------------------------




    #-------------------------------------------------------------------

    return V_new, projected_data, reconstructed_data

if __name__ == '__main__':
    # from PIL import Image
    # print("Running Image Compression using PCA")
    # image = Image.open('/Users/Rajkumar/Downloads/sample_faces/sample.jpeg')
    # image = np.array(image,dtype=np.float64)
    # global_min = np.min(image)
    # global_max = np.max(image)
    # image = (image-global_min)/(global_max-global_min)
    # X,Y,_ = image.shape
    # reconstructed_img = np.zeros([X,Y,3],dtype=np.float64)
    # _,reconstructed_img[:,:,0] = pca(image[:,:,0],image=True,retain_comp=50)
    # _,reconstructed_img[:,:,1] = pca(image[:,:,1],image=True,retain_comp=50)
    # _,reconstructed_img[:,:,2] = pca(image[:,:,2],image=True,retain_comp=50)
    # global_min = np.min(reconstructed_img)
    # global_max = np.max(reconstructed_img)
    # reconstructed_img = (reconstructed_img-global_min)/(global_max-global_min)
    # plt.figure()
    # plt.imshow(reconstructed_img)
    # plt.show()
    # print("debug...")
    _,_,_ = pca(retain_threshold=0.008,plot_basic=False)
    input()