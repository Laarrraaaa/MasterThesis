import numpy as np
from utils.utils import Utils
import matplotlib.pyplot as plt
import scipy

__author__ = "Lara Gorini"


class GaussianProcessRegressor():

    def __init__(self, cov_function, kernelparameter, noise=1e-5):
        self.cov_function = cov_function
        self.kernelparameter = kernelparameter
        self.noise = noise

    def fit(self, Xtrain, Ytrain, kernel_inv=None):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        if kernel_inv is not None:
            self.sigma_inv = kernel_inv
        # if toeplitz:
        #     self.sigma_inv = scipy.linalg.solve_toeplitz((self.sigma[:, 0], self.sigma[0, :]), np.eye(N),
        #                                                  check_finite=True)
        else:
            self.sigma = self.cov_function(Xtrain, Xtrain, self.kernelparameter)
            self.sigma += np.eye(len(self.sigma)) * self.noise
            self.L = np.linalg.cholesky(self.sigma)
            self.L_inv = np.linalg.solve(self.L, np.eye(self.L.shape[0]))
            self.sigma_inv = self.L_inv.T @ self.L_inv

            # self.sigma = self.cov_function(Xtrain, Xtrain, self.kernelparameter)
            # self.sigma += np.eye(N) * self.noise
            # self.sigma_inv = np.linalg.solve(self.sigma, np.eye(N))

    def predict(self, Xtest):  # predict unknown function values of Xtest
        C = self.cov_function(self.Xtrain, Xtest, self.kernelparameter)
        K_test = self.cov_function(Xtest, Xtest, self.kernelparameter)
        mean_predict = C.T @ self.sigma_inv @ self.Ytrain
        cov_predict = K_test - C.T @ self.sigma_inv @ C
        return mean_predict, cov_predict  # posterior mean and covariance

    def sample_cond(self, Xtest):
        mean, cov = self.predict(Xtest)
        tmp = Utils.sample_gaussian(mean, cov + np.eye(cov.shape[0]) * self.noise)
        return tmp

    def gp_figure(self, events_train, events_test, g_train, g_test):
        plt.figure()
        idx = np.argsort(events_train.flatten())
        plt.scatter(events_train.flatten()[idx], g_train.flatten()[idx], color='blue', s=20, alpha=1,
                    label='train_events')

        idx = np.argsort(events_test.flatten())
        plt.scatter(events_test.flatten()[idx], g_test.flatten()[idx], label='predicted_events', color='red', s=20,
                    alpha=1)

        all_events = np.concatenate((events_test, events_train), axis=0).flatten()
        all_g = np.concatenate((g_test, g_train), axis=0).flatten()
        idx = np.argsort(all_events)
        plt.plot(all_events[idx], all_g[idx], color='green')

        plt.legend()
        plt.show()
