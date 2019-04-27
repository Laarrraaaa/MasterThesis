import numpy as np
import matplotlib.pyplot as plt
from model.poisson_process_sampler import PoissonProcessSampler
from sampler.sampler import SGCP_Sampler
from sampler.adams_sampler import AdamsSampler
from sampler.gibb_sampler import GibbSampler
from utils.utils import Utils
from utils.plots import Plots

__author__ = "Lara Gorini"


def set_points(nb_max, dim, diff):
    nb_dim = int(np.floor(np.power(nb_max, 1 / dim)))
    points = np.zeros([dim, nb_dim])
    for d in range(dim):
        points[d, :] = np.linspace(0, diff[0, d], num=nb_dim, endpoint=True)
    points = np.array(np.meshgrid(*points.tolist()))
    points = np.array(points).reshape([dim, -1]).T
    return points


def get_nb_inducing_points(acc_events, dim):
    max_nb_ind_points = np.arange(2, len(acc_events) + 1)
    nb_dim = np.floor(np.power(max_nb_ind_points, 1. / dim))
    nb_dim = nb_dim ** dim
    nb_ind_points = np.unique(nb_dim)
    return nb_ind_points


def get_inducing_points(acc_events, dim, diff):
    nb_ind_points = get_nb_inducing_points(acc_events, dim)
    nb = len(nb_ind_points)

    print("nb inducing points %d" % nb)

    maxiter = 1000
    U = []
    inducing_points_full = []

    limit = calc_limit(acc_events, maxiter, dim)
    print('utility limit = %.2f' % limit)

    for i in range(nb):
        inducing_points = set_points(nb_ind_points[i], dim, diff)
        inducing_points_full.append(inducing_points)
        u = calc_u(inducing_points, maxiter, dim)
        U.append(u)

        tol = 0.02
        tol_li = 0.01

        if i >= 1:
            a = U[i] / U[i - 1] - 1
            li = limit / U[i] - 1
            if ((a > 0) and (a <= tol)) and ((li > 0) and (li <= tol_li)):
                break
            if u > limit:
                break

    return inducing_points_full, U


def calc_limit(acc_events, maxiter, dim):
    u = np.zeros(maxiter)
    for i in range(maxiter):
        kk = np.random.gamma(shape=5, scale=1. / 3., size=dim)
        aa = np.random.uniform(0, 20)
        kp = [aa, np.array(kk)]
        u[i] = np.trace(Utils.gaussian_kernel_matrix(acc_events, acc_events, kp))
    limit = np.mean(u)
    return limit


def calc_u(inducing_points, maxiter, dim, rand=False):
    u = np.zeros(maxiter)
    for i in range(maxiter):
        kk = np.random.gamma(shape=5, scale=1. / 3., size=dim)
        aa = np.random.uniform(0, 20)
        kp = [aa, np.array(kk)]
        C = Utils.gaussian_kernel_matrix(acc_events, inducing_points, kp)
        D = Utils.gaussian_kernel_matrix(inducing_points, inducing_points, kp)
        if rand:
            D = D + np.eye(D.shape[0]) * 1e-5
        u[i] = np.trace(C @ np.linalg.solve(D + np.eye(D.shape[0]) * 1e-5, np.eye(len(D))) @ C.T)  # TODO
    return np.mean(u)


if __name__ == "__main__":
    llambda = 30
    maxiter = 1000
    burnin = 200
    kernelparameter = [5., np.array([0.5], dtype=np.float)]
    borders = np.array([[0, 5]])
    dim = borders.shape[0]
    diff = (borders[:, 1] - borders[:, 0])[np.newaxis, :]  # (1,dim)

    max_nb_events_grid = 1000
    nb_events = None
    nb_test_sets = None

    noise = 1e-4

    events_grid = set_points(max_nb_events_grid, dim, diff)

    acc_events, events_grid, intensity_grid, _, _ = PoissonProcessSampler.inhomogeneous(
        borders, kernelparameter, SGCP_Sampler.sigmoid, Utils.gaussian_kernel_matrix,
        max_nb_events_grid, nb_events_sets=nb_test_sets, upper_bound=llambda,
        max_nb_events=nb_events)

    print('Nb acc events: %d' % acc_events.shape[0])

    adams = AdamsSampler(acc_events, maxiter, Utils.gaussian_kernel_matrix, burnin, borders,
                         events_grid, kernelparameter=kernelparameter, noise=noise)

    adams.run()

    gibbs = GibbSampler(acc_events, maxiter, Utils.gaussian_kernel_matrix, burnin, borders,
                        events_grid, kernelparameter=kernelparameter, noise=noise)
    gibbs.run()

    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 8))

    std_adams = np.std(adams.intensities[burnin:, :], axis=0)
    std_gibbs = np.std(gibbs.intensities[burnin:, :], axis=0)
    Plots.plot_1d_axis(ax1, "Gibbs Sampler", diff, acc_events, events_grid, gibbs.mean_intensities,
                       upper_bound=llambda, grid_intensity=intensity_grid, error_bar=std_gibbs)
    Plots.plot_1d_axis(ax2, "Adams Sampler", diff, acc_events, events_grid, adams.mean_intensities,
                       upper_bound=llambda, grid_intensity=intensity_grid, error_bar=std_adams, adams=True)

    inducing_points_full, U = get_inducing_points(acc_events, dim, diff)
    inducing_points = inducing_points_full[-1]

    print(inducing_points)

    print('Nb inducing points: %d' % len(inducing_points))

    gibbs_ind = GibbSampler(acc_events, maxiter, Utils.gaussian_kernel_matrix, burnin, borders,
                            events_grid, kernelparameter=kernelparameter,
                            inducing_points=inducing_points)
    gibbs_ind.run()

    std_ind = np.std(gibbs_ind.intensities[burnin:, :], axis=0)
    std_gibbs = np.std(gibbs.intensities[burnin:, :], axis=0)

    Plots.plot_1d_axis_ind(ax3, "Gibbs Sampler Inducing Points", diff, acc_events, events_grid,
                           gibbs_ind.mean_intensities,
                           upper_bound=llambda, grid_intensity=intensity_grid, inducing_points=inducing_points,
                           error_bar=std_gibbs, gibbs=True)

    fig.tight_layout()
    plt.show()