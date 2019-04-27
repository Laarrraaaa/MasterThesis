import numpy as np

__author__ = "Lara Gorini"


class SGCP_Sampler():

    def __init__(self, observed_events, maxiter, cov_function, burnin, borders, grid_events,
                 noise=1e-5, kernelparameter=None, inducing_points=None):

        self.time = 0

        self.observed_events = observed_events
        self.N = observed_events.shape[0]

        self.cov_function = cov_function

        self.maxiter = maxiter
        self.burnin = burnin
        self.noise = noise

        self.borders = borders
        self.dim = borders.shape[0]
        self.diff = (borders[:, 1] - borders[:, 0])[np.newaxis, :]  # (1,dim)
        self.vol = np.prod(self.diff)

        if kernelparameter is None:
            self.sample_kernel_parameter = True
            #self.kernelparameter = [np.random.uniform(0, 20), np.random.gamma(4., scale=1 / 3., size=self.dim)]
            self.kernelparameter = [15, np.random.gamma(4., scale=1 / 3., size=self.dim)]
        else:
            self.sample_kernel_parameter = False
            self.kernelparameter = kernelparameter

        self.inducing_points = inducing_points

        self.alpha = 4.
        self.beta = 2. * self.vol / self.N

        self.grid_events = grid_events

        ##### INITS ##################

        self.upper_bound = np.random.rand(1) * 2 ** 8
        self.M = np.random.randint(1, 2 ** 8, size=None)

        #### RESULTS ############

        # self.mean_intensities = np.zeros(self.grid_events.shape[0])
        self.llambdas = np.zeros(self.maxiter)
        self.intensities = np.zeros([self.maxiter, self.grid_events.shape[0]])
        self.latent_M = np.zeros(self.maxiter)

        self.events = []
        self.gaussians = []

    @staticmethod
    def sigmoid(G):
        return 1. / (1. + np.exp(-G))
