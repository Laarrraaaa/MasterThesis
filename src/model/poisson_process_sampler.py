import numpy as np
from utils.utils import Utils
import matplotlib.pyplot as plt
from sampler.gaussian_process_regressor import GaussianProcessRegressor

__author__ = 'Lara Gorini'

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


class PoissonProcessSampler():

    # events always in range [0, diff], borders are rescaled to start at zero
    @staticmethod
    def inhomogeneous(borders, kernelparameter, intensity_function, cov_function, nb_total_events_grid,
                      nb_events_sets=None, upper_bound=None, max_nb_events=None):

        dim = borders.shape[0]
        diff = (borders[:, 1] - borders[:, 0])[np.newaxis, :]  # (1,dim)
        vol = np.prod(diff)

        if nb_events_sets is not None:
            nb_events_sets_xx = nb_events_sets + 1  # plus accepted events
        else:
            nb_events_sets_xx = 1

        if (upper_bound is None) and (max_nb_events is None):
            raise ValueError('upper_bound or max_nb_events must not be none.')
        elif upper_bound is None:
            upper_bound = max_nb_events // vol

        nb_expected_events = vol * upper_bound * nb_events_sets_xx

        nb_events = np.random.poisson(lam=nb_expected_events, size=None)  # nb_events
        events = np.random.uniform(0, 1, size=(nb_events, dim)) * diff

        print('nb expected events: %d' % nb_expected_events)
        print('nb events: %d' % events.shape[0])

        nb_events_grid = int(np.floor(np.power(nb_total_events_grid, 1 / dim)))
        events_grid = np.zeros([dim, nb_events_grid])
        for d in range(dim):
            events_grid[d, :] = np.linspace(0, diff[0, d], num=nb_events_grid, endpoint=True)
        events_grid = np.array(np.meshgrid(*events_grid.tolist()))
        events_grid = np.array(events_grid).reshape([dim, -1]).T

        print('nb_events_grid: %d' % events_grid.shape[0])

        all_events = np.concatenate((events_grid, events), axis=0)  # (nb_events_grid+nb_events, dim)

        K = cov_function(all_events, all_events, kernelparameter)
        G = Utils.sample_gaussian(np.zeros([all_events.shape[0], 1]), K + np.eye(all_events.shape[0]) * 1e-5).flatten()

        # sample accepted events
        R = np.random.uniform(0, upper_bound, size=nb_events)
        intensity = upper_bound * intensity_function(G[events_grid.shape[0]:])  # intensity for events_sets
        thinned_idx = R < intensity

        event_set_id = np.random.randint(0, nb_events_sets_xx, nb_events)  # (nb_events,)
        event_set_id_thinned = event_set_id[thinned_idx]
        events_sets = []
        intensity_sets = []
        for i in range(nb_events_sets_xx):
            idx = event_set_id_thinned == i  # which events belong to set i
            events_sets.append(events[thinned_idx][idx])  # take the thinned events only
            intensity_sets.append(intensity[thinned_idx][idx])

        # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=False, sharey=False, figsize=(7, 7))
        # ax1.vlines(events, [0], np.full(events.shape[0], 0.75), linewidth=0.5)
        # ax1.axhline(upper_bound, linewidth=0.8, color='black', linestyle=':')
        # ax1.set_xlim([borders[0, 0], borders[0, 1]])
        # ax1.set_ylim([0, upper_bound + 1])
        # ax1.set_xlabel(r'$x$' + '\n' + r'(a)')
        # ax1.set_ylabel(r'$\varphi(x)$')
        # ax1.set_yticks([0, upper_bound])
        # ax1.set_xticks([0, borders[0, 1]])
        #
        # idx = np.argsort(all_events.flatten())
        # ax2.vlines(events, [0], np.full(events.shape[0], 0.75), linewidth=0.5)
        # ax2.axhline(upper_bound, linewidth=0.8, color='black', linestyle=':')
        # ax2.plot(all_events[idx], upper_bound * intensity_function(G[idx]), linewidth=0.8, color='black')
        # ax2.set_xlim([borders[0, 0], borders[0, 1]])
        # ax2.set_ylim([0, upper_bound + 1])
        # ax2.set_xlabel(r'$x$' + '\n' + r'(b)')
        # ax2.set_ylabel(r'$\varphi(x)$')
        # ax2.set_yticks([0, upper_bound])
        # ax2.set_xticks([0, borders[0, 1]])
        #
        # ax3.vlines(events, [0], np.full(events.shape[0], 0.75), linewidth=0.5)
        # ax3.axhline(upper_bound, linewidth=0.8, color='black', linestyle=':')
        # ax3.plot(all_events[idx], upper_bound * intensity_function(G[idx]), linewidth=0.8, color='black')
        # ax3.scatter(events, R, s=2.5, color='black')
        # ax3.set_xlim([borders[0, 0], borders[0, 1]])
        # ax3.set_ylim([0, upper_bound + 1])
        # ax3.set_xlabel(r'$x$' + '\n' + r'(c)')
        # ax3.set_ylabel(r'$\varphi(x)$')
        # ax3.set_yticks([0, upper_bound])
        # ax3.set_xticks([0, borders[0, 1]])
        #
        # ax4.vlines(events_sets[0], [0], np.full(events_sets[0].shape[0], 0.75), linewidth=0.5)
        # ax4.axhline(upper_bound, linewidth=0.8, color='black', linestyle=':')
        # ax4.plot(all_events[idx], upper_bound * intensity_function(G[idx]), linewidth=0.8, color='black')
        # ax4.set_xlim([borders[0, 0], borders[0, 1]])
        # ax4.set_ylim([0, upper_bound + 1])
        # ax4.set_yticks([0, upper_bound])
        # ax4.set_xticks([0, borders[0, 1]])
        # ax4.set_xlabel(r'$x$' + '\n' + r'(d)')
        # ax4.set_ylabel(r'$\varphi(x)$')
        #
        # plt.tight_layout()
        # plt.savefig('/home/lara/Desktop/Masterarbeit/tex/data/adams_generative_model.pdf')
        # plt.show()

        if nb_events_sets is None:
            return events_sets[0], events_grid, upper_bound * intensity_function(
                G[:events_grid.shape[0]]), upper_bound, None
        else:
            return events_sets[0], events_grid, upper_bound * intensity_function(
                G[:events_grid.shape[0]]), upper_bound, events_sets[1:]
