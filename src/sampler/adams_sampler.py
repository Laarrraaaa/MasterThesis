import numpy
import time
from sampler.sampler import SGCP_Sampler
from utils.utils import Utils

__author__ = "Christian Donner"


class AdamsSampler(SGCP_Sampler):

    def __init__(self, *args, **kwargs):
        super(AdamsSampler, self).__init__(*args, **kwargs)

        # Position of all events (first N are the actual observed ones)
        self.X_all = numpy.empty([self.N + self.M, self.dim])
        self.X_all[:self.N] = self.observed_events
        self.X_all[self.N:] = numpy.random.rand(self.M, self.dim)
        self.K = self.cov_func(self.X_all, self.X_all)
        self.L = numpy.linalg.cholesky(self.K + self.noise * numpy.eye(self.K.shape[0]))
        self.L_inv = numpy.linalg.solve(self.L, numpy.eye(self.L.shape[0]))
        self.K_inv = self.L_inv.T.dot(self.L_inv)

        L_X = numpy.linalg.cholesky(self.K[:self.N, :self.N] + self.noise * numpy.eye(self.N))
        L_X_inv = numpy.linalg.solve(L_X, numpy.eye(self.N))
        self.K_X_inv = L_X_inv.T.dot(L_X_inv)
        # Initial values of function g
        self.g = numpy.zeros([self.N + self.M])
        # Probability of insertion or deletion proposal
        self.pM = .5
        self.M_iterations = 10

    def run(self):

        print('Starting Adams')

        start = time.time()

        for i in range(self.maxiter):

            if i == 1:
                loop_start = time.time()

            if i == 2:
                print('Approximately %.2f min to go' % (loop * self.maxiter / 60))

            if (i > 0) and (i % 50 == 0):
                print(i)

            self.sample_g()
            self.sample_M()
            self.sample_locations()
            self.sample_upper_bound()

            self.llambdas[i] = self.upper_bound
            self.latent_M[i] = self.M
            self.intensities[i, :] = self.calc_intensities()
            self.events.append(self.X_all)
            self.gaussians.append(self.g)

            if i == 1:
                loop = time.time() - loop_start

        self.time = (time.time() - start) / 60
        print('Done in %.2f min' % self.time)
        self.mean_intensities = numpy.mean(self.intensities[self.burnin:], axis=0)

    def cov_func(self, x, x_prime):
        """ Computes the covariance functions between x and x_prime.
        :param x: numpy.ndarray [num_points x D]
            Contains coordinates for points of x
        :param x_prime: numpy.ndarray [num_points_prime x D]
            Contains coordinates for points of x_prime
        :return: numpy.ndarray [num_points x num_points_prime]
            Kernel matrix.
        """
        return self.cov_function(x, x_prime, self.kernelparameter)

    def sample_M(self):
        """ Samples number of events.
        """

        propose_insertion = numpy.random.rand(self.M_iterations) < self.pM
        rand_numbers_accept = numpy.random.rand(self.M_iterations)

        for i in range(self.M_iterations):
            # Insertion
            if propose_insertion[i]:
                xprime = numpy.random.rand(1, self.dim) * self.diff
                gprime = self.sample_from_cond_GP(xprime)
                a_ins = (1. - self.pM) * self.upper_bound * self.vol / (
                        (self.M + 1.) * self.pM * (1. + numpy.exp(gprime)))
                if rand_numbers_accept[i] < a_ins:
                    self.M += 1
                    self.X_all = numpy.concatenate([self.X_all, xprime])
                    self.g = numpy.concatenate([self.g, gprime])
                    self.update_kernels()
            # Deletion
            elif self.M > 0:
                del_idx = numpy.random.randint(self.N, self.N + self.M)
                gdel = self.g[del_idx]
                a_del = self.M * self.pM * (1. + numpy.exp(gdel)) / ((1. - self.pM) * self.vol * self.upper_bound)
                if rand_numbers_accept[i] < a_del:
                    self.M -= 1
                    self.X_all = numpy.delete(self.X_all, del_idx, 0)
                    self.g = numpy.delete(self.g, del_idx, 0)
                    self.update_kernels()

    def sample_locations(self):

        theta2 = self.kernelparameter[1]
        proposed_X = self.X_all[self.N:] + numpy.random.randn(self.M, self.dim) * theta2[numpy.newaxis, :]

        # proposed_X = self.X_all[self.N:] + numpy.random.randn(self.L, self.dim) * self.kernelparameter

        rand_num_accept = numpy.random.rand(self.M)

        inregion1 = numpy.equal(numpy.sum(proposed_X < 0, axis=1), 0)
        inregion2 = numpy.equal(numpy.sum((self.diff - proposed_X) < 0, axis=1), 0)  ############### !!!! ############
        inregion = numpy.where(numpy.logical_and(inregion1, inregion2))[0]

        for im in inregion:
            xprime = proposed_X[im]
            gprime = self.sample_from_cond_GP(numpy.array([xprime]))
            a_loc = (1. + numpy.exp(self.g[im + self.N])) / (1. + numpy.exp(gprime))
            if rand_num_accept[im] < a_loc:
                self.X_all[im + self.N] = xprime
                self.g[im + self.N] = gprime
                self.update_kernels_loc(im, xprime)

        self.update_kernels()

    def update_kernels_loc(self, im, xprime):
        new_row = self.cov_func(self.X_all, xprime[numpy.newaxis, :])
        self.K[im + self.N] = new_row[:, 0]
        self.K[:, im + self.N] = new_row[:, 0]
        B = self.K[:self.N, self.N:]
        C = self.K[self.N:, self.N:]
        self.K_inv = self.invert_block_matrix(self.K_X_inv, B, C)

    def sample_upper_bound(self):
        self.upper_bound = numpy.random.gamma(self.N + self.M + self.alpha, 1 / (self.vol + self.beta))

    def sample_g(self):
        r_init = numpy.random.randn(self.g.shape[0])
        H_init = self.compute_Hamiltonian(r_init)
        proposed_g, proposed_r = self.leap_frog_integration(r_init)
        H_proposed = self.compute_Hamiltonian(proposed_r, proposed_g)
        a_funcg = numpy.exp(H_init - H_proposed)
        rand_num_accept = numpy.random.rand(1)

        if rand_num_accept < a_funcg:
            self.g = proposed_g

    def compute_Hamiltonian(self, r, g=None):
        if g is None:
            g = self.g
        E = .5 * g.T.dot(self.K_inv.dot(g)) + \
            numpy.sum(numpy.log(1. + numpy.exp(-g[:self.N]))) + \
            numpy.sum(numpy.log(1. + numpy.exp(g[self.N:])))
        K = .5 * numpy.sum(r ** 2)
        return E + K

    def leap_frog_integration(self, r, g=None, step_size=1e-2):
        if g is None:
            g = numpy.copy(self.g)
        gtilde = self.L_inv.dot(g)
        step_size += 1e-4 * numpy.random.randn()

        rand_num = numpy.random.rand(1)
        if rand_num < .5:
            step_size *= -1

        num_steps = int(numpy.ceil(1. / numpy.abs(step_size)))

        for isteps in range(num_steps):
            dgtilde = self.compute_gradient_gtilde(gtilde)
            r += .5 * step_size * dgtilde
            gtilde -= step_size * r
            dgtilde = self.compute_gradient_gtilde(gtilde)
            r += .5 * step_size * dgtilde
        proposed_g = self.L.dot(gtilde)
        return proposed_g, r

    def compute_gradient_gtilde(self, gtilde):
        g = self.L.dot(gtilde)
        v = numpy.zeros([self.M + self.N])
        v[:self.N] = - 1. / (1. + numpy.exp(g[:self.N]))
        v[self.N:] = 1. / (1. + numpy.exp(-g[self.N:]))
        dgtilde = gtilde + v.dot(self.L)
        return dgtilde

    def update_kernels(self):
        self.K = self.cov_func(self.X_all, self.X_all)
        self.L = numpy.linalg.cholesky(self.K + self.noise * numpy.eye(self.K.shape[0]))
        self.L_inv = numpy.linalg.solve(self.L, numpy.eye(self.L.shape[0]))
        self.K_inv = self.L_inv.T.dot(self.L_inv)

    def sample_from_cond_GP(self, xprime):
        k = self.cov_func(self.X_all, xprime)
        mean = k.T.dot(self.K_inv.dot(self.g))
        kprimeprime = self.cov_func(xprime, xprime)
        cov = kprimeprime - k.T.dot(self.K_inv.dot(k))
        if len(cov) == 1:
            var = cov.diagonal()
            gprime = mean + numpy.sqrt(var) * numpy.random.randn(xprime.shape[0])
        else:
            gprime = Utils.sample_gaussian(mean[:, numpy.newaxis], cov + numpy.eye(len(cov)) * self.noise).flatten()
        return gprime

    def invert_block_matrix(self, A_inv, B, C):
        """ Inverts a matrix of the form
                [[A,  B]
                 [B', C]].
        """

        A_dim, C_dim = A_inv.shape[0], C.shape[0]
        Mat_dim = A_dim + C_dim
        M_inv = C - B.T.dot(A_inv.dot(B))
        LM_inv = numpy.linalg.cholesky(M_inv + 1e-3 * numpy.eye(C_dim))
        LM = numpy.linalg.solve(LM_inv, numpy.eye(LM_inv.shape[0]))
        M = LM.T.dot(LM)

        Mat_inv = numpy.empty([Mat_dim, Mat_dim])
        Mat_inv[:A_dim, :A_dim] = A_inv + A_inv.dot(B.dot(M.dot(B.T.dot(A_inv))))
        Mat_inv[:A_dim, A_dim:] = - A_inv.dot(B.dot(M))
        Mat_inv[A_dim:, :A_dim] = Mat_inv[:A_dim, A_dim:].T
        Mat_inv[A_dim:, A_dim:] = M

        return Mat_inv

    def calc_intensities(self):
        g_pred = self.sample_from_cond_GP(self.grid_events)
        return self.upper_bound * SGCP_Sampler.sigmoid(g_pred)