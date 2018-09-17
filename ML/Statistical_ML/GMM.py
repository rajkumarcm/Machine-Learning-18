"""
Author: Rajkumar Conjeevaram Mohan
Email: rajkumarcm@yahoo.com
Program: Clustering using Gaussian Mixture Model EM
"""

import numpy as np
import data
from scipy.stats import multivariate_normal as m_norm
import matplotlib.pyplot as plt
import progressbar as pb

def normpdf(x, mu, sigma):
    # x_c = x - mu
    # M = x_c.shape[1]
    # num = np.exp(-1/2*np.dot(x_c,np.dot(np.linalg.inv(sigma),x_c.T)))
    # den = np.sqrt(((2 * np.pi)**M) * np.linalg.det(sigma))
    # return num/den
    return m_norm(mean=mu,cov=sigma,allow_singular=True).pdf(x)

def GMM(X=None,Y=None,epochs=10):

    if X is None:
        # Import the data
        X,Y = data.load_data("breast_cancer_kaggle.csv","\n",",",1)
        # DELETE THIS WHEN DIFFERENT DATASET IS USED----------------
        X = X[:,1:]

    N,M = X.shape

    # Initialise parameters
    classes = np.unique(Y)
    n_classes = len(classes)
    sample_y = np.random.randint(0,n_classes,size=[N])
    means = np.zeros(shape=[n_classes,M])
    cov = np.zeros([n_classes,M,M])
    for k in range(n_classes):
        temp = X[sample_y == k]
        means[k] = np.mean(temp, axis=0)
        cov[k] = np.dot((temp-means[k]).T,(temp-means[k]))/(temp.shape[0]-1)
    # P_C = np.random.random((n_classes))
    # P_C /= np.sum(P_C)
    P_C = np.array([0.6,0.4])
    P_C_X = np.zeros((N,n_classes))

    with pb.ProgressBar(max_value=epochs) as bar:
        for epoch in range(epochs):
            # Expectation
            for i in range(N):
                temp = np.zeros((n_classes))
                for k in range(n_classes):
                    temp[k] = P_C[k] * normpdf(X[i],means[k],cov[k])
                P_C_X[i] = temp/np.sum(temp)
            ll = np.sum(np.log(np.sum(P_C_X,axis=1)))
            print("Log-Likelihood: %.2f"%(ll))

            # Maximization
            for k in range(n_classes):
                old_means = np.copy(means)
                P_C[k] = np.sum(P_C_X[:,k])/N
                den = np.sum(P_C_X[:,k])
                means[k] = np.zeros(shape=[1,M])
                cov_temp = np.zeros([M,M])
                for i in range(N):
                    means[k] += P_C_X[i,k] * X[i]
                    temp = X[i] - old_means[k]
                    cov_temp += (P_C_X[i,k] * np.dot(temp.T,temp))/(den)
                means[k] /= den
                cov[k] = cov_temp
            bar.update(value=epoch)
            if epoch%50 == 0:
                print("checkpoint...")

    pred_labels = np.argmax(P_C_X,axis=1)

    plt.figure()
    plt.title("Original Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid()
    plt.tight_layout()
    for label, marker, color in zip(
            classes, ('^', 's', 'o'), ('blue', 'red', 'green')):
        plt.scatter(X[:, 0].real[Y == label],
                    X[:, 1].real[Y == label],
                    c=color, marker=marker, alpha=0.5)

    plt.figure()
    plt.title("GMM Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid()
    plt.tight_layout()
    for label, marker, color in zip(
            np.unique(pred_labels), ('^', 's', 'o'), ('blue', 'red', 'green')):
        plt.scatter(X[:, 0].real[pred_labels == label],
                    X[:, 1].real[pred_labels == label],
                    c=color, marker=marker, alpha=0.5)
    plt.show()
    input()


# X1 = np.random.normal(loc=100,scale=15,size=[50,10])
# X2 = np.random.normal(loc=200,scale=50,size=[50,10])
# X = np.vstack((X1,X2))
mean1 = np.random.normal(20,0.1,size=[1,10])
mean2 = np.random.normal(10,0.1,size=[1,10])
means = np.vstack((mean1,mean2))

cov = np.zeros([2,10,10])
cov[0] = np.eye(10,10) * np.random.random()
cov[1] = np.eye(10,10) * np.random.random()

X1 = np.random.multivariate_normal(means[0],cov[0],size=[50])
X2 = np.random.multivariate_normal(means[1],cov[1],size=[50])
X = np.vstack((X1,X2))

Y1 = np.zeros([50,1])
Y2 = np.ones([50,1])
Y = np.vstack((Y1,Y2))

GMM(X,Y,epochs=100)