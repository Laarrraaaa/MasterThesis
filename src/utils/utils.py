import numpy as np
from scipy.spatial.distance import cdist

__author__ = "Lara Gorini"


class Utils():

    @staticmethod
    def create_diff_matrix(XA, XB):
        x = np.array(XA)
        y = np.array(XB)
        diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]  # (NA, NB, D)
        return diff

    @staticmethod
    def gaussian_kernel_matrix(XA, XB, parameter):
        theta2 = parameter[1]
        theta1 = parameter[0]

        diff = Utils.create_diff_matrix(XA, XB)
        param = np.repeat(theta2[np.newaxis, :], XA.shape[0], axis=0)
        param = np.repeat(param[:, np.newaxis, :], XB.shape[0], axis=1)
        kernel_matrix = np.sum(diff ** 2 / param ** 2, axis=2)

        kernel_matrix = theta1 * np.exp(- 0.5 * kernel_matrix)

        # tmp = theta1 * np.exp(-1. / 2 * cdist(XA, XB, 'minkowski', p=2, w=1 / theta2 ** 2) ** 2)
        # tt = np.allclose(tmp, kernel_matrix, atol=1e-6, rtol=0.0)
        # print(tt)

        return kernel_matrix

    # @staticmethod
    # def gaussian_kernel_matrix_exp(XA, XB, parameter):
    #     theta1 = parameter[0]
    #     theta2 = np.exp(parameter[1])
    #     param = [theta1, theta2]
    #     # print('param: %f' % theta2[0])
    #     K = Utils.gaussian_kernel_matrix(XA, XB, param)
    #     return K + np.eye(K.shape[0]) * 1e-5

    @staticmethod
    def sample_gaussian(mean, cov):
        # mean (N,1)
        # cov (N,N)
        N = mean.shape[0]
        L = np.linalg.cholesky(cov)
        Z = np.random.normal(0, 1, size=(N, 1))
        sample = mean + L @ Z
        return sample
