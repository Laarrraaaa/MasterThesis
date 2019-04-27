import numpy as np
from pypolyagamma import PyPolyaGamma
from utils.utils import Utils
import time
from sampler.sampler import SGCP_Sampler

__author__ = "Lara Gorini"


class GibbSampler(SGCP_Sampler):

    def __init__(self, *args, **kwargs):
        super(GibbSampler, self).__init__(*args, **kwargs)

        self.pg = PyPolyaGamma(seed=np.random.randint(2 ** 16, size=None))

    def run(self):

        print('Starting Gibbs')

        latent_events = np.random.rand(self.M, self.dim) * self.diff
        latent_marks = np.random.rand(self.M, 1) * 2 ** 10  # distribute on space
        marks = np.random.rand(self.N, 1) * 2 ** 10  # distribute on space

        start = time.time()

        for k in range(self.maxiter):

            if k == 1:
                loop_start = time.time()
            if k == 2:
                print('Approximately %.2f min to go' % (loop * self.maxiter / 60))

            if self.inducing_points is not None:

                if k == 0:
                    self.events_base = self.inducing_points
                    self.K = self.cov_function(self.events_base, self.events_base, self.kernelparameter)
                    self.K += np.eye(len(self.K)) * self.noise
                    self.L = np.linalg.cholesky(self.K)
                    self.L_inv = np.linalg.solve(self.L, np.eye(self.L.shape[0]))
                    self.K_inv = self.L_inv.T @ self.L_inv

                self.sample_upper_bound(latent_marks.shape[0])
                self.sample_gaussian_induced(latent_events, marks, latent_marks)

                if self.sample_kernel_parameter:
                    if (k % 10) == 0:
                        self.sample_kernelparameter()

                intensity = self.sample_results()

                latent_events, g_M, g_N = self.sample_latent_events_induced()
                latent_marks = self.sample_latent_marks(g_M)
                marks = self.sample_marks(g_N)

            else:

                self.events_base = np.concatenate((self.observed_events, latent_events), axis=0)
                self.K = self.cov_function(self.events_base, self.events_base, self.kernelparameter)
                self.K += np.eye(len(self.K)) * self.noise
                self.L = np.linalg.cholesky(self.K)
                self.L_inv = np.linalg.solve(self.L, np.eye(self.L.shape[0]))
                self.K_inv = self.L_inv.T @ self.L_inv

                self.sample_upper_bound(latent_marks.shape[0])
                self.sample_gaussian(marks, latent_marks)

                if self.sample_kernel_parameter:
                    if (k % 10) == 0:
                        self.sample_kernelparameter()  # updates the kernels

                intensity = self.sample_results()

                latent_events, g_M = self.sample_latent_events()
                latent_marks = self.sample_latent_marks(g_M)
                marks = self.sample_marks(np.array(self.g[:self.N, :]))

            if ((k > 0) or (k == self.maxiter - 1)) and (k % 50 == 0):
                if self.inducing_points is not None:
                    print('%d   with inducing points' % k)
                else:
                    print(k)

            self.llambdas[k] = self.upper_bound
            self.latent_M[k] = latent_marks.shape[0]
            self.intensities[k, :] = intensity
            # self.log_likelihoods[k, :] = log_likelihood

            if k == 1:
                loop = time.time() - loop_start

        self.time = (time.time() - start) / 60
        print('Done in %.2f min' % self.time)
        self.mean_intensities = np.mean(self.intensities[self.burnin:], axis=0)

    ######################################################################

    def sample_gaussian_induced(self, latent_events, marks, latent_marks):
        L_ind = len(self.inducing_points)
        kN = self.cov_function(self.inducing_points, self.observed_events, self.kernelparameter)
        kM = self.cov_function(self.inducing_points, latent_events, self.kernelparameter)
        BN = kN[np.newaxis, ::] * kN[::, np.newaxis]  # (L,L,N)
        BM = kM[np.newaxis, ::] * kM[::, np.newaxis]  # (L,L,M)
        wN = np.repeat(marks, L_ind, axis=1)
        wN = np.repeat(wN[:, :, np.newaxis], L_ind, axis=2).T
        wM = np.repeat(latent_marks, L_ind, axis=1)
        wM = np.repeat(wM[:, :, np.newaxis], L_ind, axis=2).T

        B = np.sum(BN * wN, axis=2) + np.sum(BM * wM, axis=2)
        BLinv = np.linalg.solve(B + self.K, np.eye(L_ind))
        sigmaL = self.K @ BLinv @ self.K
        muL = 0.5 * self.K @ BLinv @ (np.sum(kN, axis=1, keepdims=True) - np.sum(kM, axis=1, keepdims=True))
        self.g = Utils.sample_gaussian(muL, sigmaL)  # + np.eye(L_ind) * self.noise)

    def sample_latent_events_induced(self):
        xx = 0
        while (xx == 0):
            latent_events, g_J, g_N = self.sample_latent_process_induced()
            xx = len(latent_events)
        return latent_events, g_J, g_N

    def sample_latent_process_induced(self):
        J = np.random.poisson(lam=self.vol * self.upper_bound, size=None)  # nb_events
        events = np.random.rand(J, self.dim) * self.diff
        g = self.sample_cond(np.concatenate((events, self.observed_events), axis=0))
        g_J = np.array(g[:len(events)])
        g_N = np.array(g[len(events):])
        R = np.random.rand(J) * self.upper_bound
        idx = R < self.upper_bound * SGCP_Sampler.sigmoid(-g_J.flatten())
        acc_events = events[idx, :]
        return acc_events, g_J[idx, :], g_N

    ######################################################################

    def sample_gaussian(self, marks, latent_marks):
        M = latent_marks.shape[0]
        marks_concat = np.concatenate((marks, latent_marks), axis=0)
        sigma = np.diag(1. / marks_concat.flatten())
        sigma_NM = sigma - sigma @ np.linalg.solve(sigma + self.K, np.eye(self.N + M)) @ sigma
        u = np.concatenate((np.full((self.N, 1), 1. / 2, ), np.full((M, 1), -1. / 2)), axis=0)
        mean_NM = sigma_NM @ u
        self.g = Utils.sample_gaussian(mean_NM, sigma_NM)  # + np.eye(sigma_NM.shape[0]) * self.noise)

    def sample_latent_process(self):
        J = np.random.poisson(lam=self.vol * self.upper_bound, size=None)  # nb_events
        events = np.random.rand(J, self.dim) * self.diff
        g_J = self.sample_cond(events)
        R = np.random.rand(J) * self.upper_bound
        idx = R < self.upper_bound * SGCP_Sampler.sigmoid(-g_J.flatten())
        acc_events = events[idx, :]
        return acc_events, g_J[idx, :]

    def sample_latent_events(self):
        xx = 0
        while (xx == 0):
            latent_events, g_J = self.sample_latent_process()
            xx = len(latent_events)
        return latent_events, g_J

    ######################################################################

    def sample_upper_bound(self, M):
        self.upper_bound = np.random.gamma(shape=self.alpha + M + self.N, scale=1. / (self.beta + self.vol))

    def sample_latent_marks(self, g_M):
        M = g_M.shape[0]
        latent_marks = np.zeros([M, 1])
        for i in range(M):
            latent_marks[i, :] = self.pg.pgdraw(1, g_M[i, :])
        return latent_marks

    def sample_kernelparameter(self):
        prop = np.random.randn(self.dim + 1)
        alpha = np.exp(np.log(self.kernelparameter[0]) + prop[0] * 0.05)
        beta = np.exp(np.log(self.kernelparameter[1]) + prop[1:] * 0.05)
        proposal = [alpha, beta]
        K = self.cov_function(self.events_base, self.events_base, proposal)
        K += np.eye(K.shape[0]) * self.noise
        L = np.linalg.cholesky(K)
        L_inv = np.linalg.solve(L, np.eye(L.shape[0]))
        K_inv = L_inv.T @ L_inv
        prop = - np.sum(np.log(L.diagonal())) - 0.5 * self.g.T @ K_inv @ self.g
        old = - np.sum(np.log(self.L.diagonal())) - 0.5 * self.g.T @ self.K_inv @ self.g
        A = min(0, np.asscalar(prop - old))
        u = np.log(np.random.rand())
        if u < A:
            self.K = K
            self.L = L
            self.L_inv = L_inv
            self.K_inv = K_inv
            self.kernelparameter = proposal
            print(self.kernelparameter)

    def predict(self, Xtest):  # predict unknown function values of Xtest
        C = self.cov_function(self.events_base, Xtest, self.kernelparameter)
        K_test = self.cov_function(Xtest, Xtest, self.kernelparameter)
        mean_predict = C.T @ self.K_inv @ self.g
        cov_predict = K_test - C.T @ self.K_inv @ C
        return mean_predict, cov_predict  # posterior mean and covariance

    def sample_cond(self, Xtest):
        mean, cov = self.predict(Xtest)
        tmp = Utils.sample_gaussian(mean, cov + np.eye(cov.shape[0]) * self.noise)
        return tmp

    def sample_marks(self, g_N):
        marks = np.zeros([self.N, 1])
        for i in range(self.N):
            marks[i, :] = self.pg.pgdraw(1, g_N[i, :])
        return marks

    def sample_results(self):
        self.events.append(self.events_base)
        self.gaussians.append(self.g)
        g = self.sample_cond(self.grid_events)
        return self.upper_bound * SGCP_Sampler.sigmoid(g.flatten())
