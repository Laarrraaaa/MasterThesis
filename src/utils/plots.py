import numpy as np
import matplotlib.pyplot as plt

SMALL_SIZE = 12
MEDIUM_SIZE = 16
plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes

plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)

plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels

plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

__author__ = "Lara Gorini"


class Plots():
    # cmap_name = 'Spectral'
    cmap_name = 'jet'
    colors = plt.cm.jet(np.linspace(0, 1, 2))
    # color_truth = 'blue'
    color_truth = colors[0]  # blue
    color_show = colors[1]  # red
    # color_show = 'red'
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    @staticmethod
    def plot_1d_axis_ind(axis, label, diff, observed_events, grid_events, mean_intensity_gibbs,
                         upper_bound=None, grid_intensity=None, error_bar=None, inducing_points=None, gibbs=False):

        if gibbs:
            color = Plots.color_truth
        else:
            color = Plots.color_show

        vmax = mean_intensity_gibbs.max()
        if grid_intensity is not None:
            vmax = max(vmax, grid_intensity.max())
        if upper_bound is not None:
            vmax = max(vmax, upper_bound)
        if error_bar is not None:
            y2 = mean_intensity_gibbs + error_bar
            y1 = mean_intensity_gibbs - error_bar
            vmax = max(vmax, y2.max())

        vmax = np.ceil(vmax)
        tmp = (vmax + 1) / 10

        axis.vlines(observed_events, [0], np.full(observed_events.shape[0], tmp), linewidth=0.5, color='darkgrey',
                    alpha=1)

        if upper_bound is not None:
            axis.axhline(upper_bound, linewidth=0.8, color='black', linestyle=':')
        if grid_intensity is not None:
            axis.plot(grid_events, grid_intensity, color='black', linestyle=':', linewidth=1)
        axis.plot(grid_events, mean_intensity_gibbs, color=color, linewidth=0.8)

        axis.set_title(r'N=%d' % len(observed_events))
        if inducing_points is not None:
            axis.set_title(r'L=%d/N=%d' % (len(inducing_points), len(observed_events)))
        axis.set_xlim([0, diff[0, 0]])
        axis.set_ylim([0, vmax + tmp])
        axis.set_xticks([0, diff[0, 0]])
        if upper_bound is not None:
            axis.set_yticks([0, upper_bound])
        else:
            axis.set_yticks([0, vmax])
        #axis.set_ylabel(r'$\varphi(x)$')
        axis.set_xlabel(label)

        if error_bar is not None:
            axis.fill_between(grid_events.flatten(), y1, y2, where=y1 <= y2, facecolor=color, interpolate=True,
                              alpha=0.2)

    @staticmethod
    def plot_1d_axis(axis, label, diff, observed_events, grid_events, mean_intensity_gibbs, upper_bound=None,
                     grid_intensity=None, error_bar=None, adams=False):

        if adams:
            color = Plots.color_truth
        else:
            color = Plots.color_show

        vmax = mean_intensity_gibbs.max()
        if grid_intensity is not None:
            vmax = max(vmax, grid_intensity.max())
        if upper_bound is not None:
            vmax = max(vmax, upper_bound)
        if error_bar is not None:
            y2 = mean_intensity_gibbs + error_bar
            y1 = mean_intensity_gibbs - error_bar
            vmax = max(vmax, y2.max())

        vmax = np.ceil(vmax)
        tmp = (vmax + 1) / 10

        axis.vlines(observed_events, [0], np.full(observed_events.shape[0], tmp), linewidth=0.5,
                    color='darkgrey')

        if upper_bound is not None:
            axis.axhline(upper_bound, linewidth=0.8, color='black', linestyle=':')

        if grid_intensity is not None:
            axis.plot(grid_events, grid_intensity, color='black', linestyle=':', linewidth=1)

        axis.plot(grid_events, mean_intensity_gibbs, color=color, linewidth=0.8)

        axis.set_xlim([0, diff[0, 0]])
        axis.set_ylim([0, vmax + tmp])
        axis.set_xticks([0, diff[0, 0]])
        if upper_bound is not None:
            axis.set_yticks([0, upper_bound])
        else:
            axis.set_yticks([0, vmax])
        #axis.set_ylabel(r'$\varphi(x)$')
        axis.set_xlabel(r'$x$' + '\n' + label)

        if error_bar is not None:
            axis.fill_between(grid_events.flatten(), y1, y2, where=y1 <= y2, facecolor=color, interpolate=True,
                              alpha=0.2)

        axis.set_title(r'N=%d' % len(observed_events))

    @staticmethod
    def plot_2d_axis_ind(axis, label, diff, vmax, observed_events, grid_events, mean_intensity, inducing_points=None,
                         ground_truth=False, plot_observed_events=False):

        nb_events_grid = int(np.sqrt(grid_events.shape[0]))
        xx = np.array(grid_events[:, 0]).reshape([nb_events_grid, nb_events_grid])
        yy = np.array(grid_events[:, 1]).reshape([nb_events_grid, nb_events_grid])

        zz_gibbs = mean_intensity.reshape([nb_events_grid, nb_events_grid])

        levels = np.linspace(0, vmax, num=100, endpoint=True, dtype=np.float)

        tmp = axis.contourf(xx, yy, zz_gibbs, vmin=0, vmax=zz_gibbs.max(), levels=levels,
                            cmap=plt.get_cmap(Plots.cmap_name))

        # axis.set_xlim([xx.min() - diff[0, 0] / 50, xx.max() + diff[0, 0] / 50])
        # axis.set_ylim([yy.min() - diff[0, 1] / 50, yy.max() + diff[0, 1] / 50])
        axis.set_xlim([xx.min(), xx.max()])
        axis.set_ylim([yy.min(), yy.max()])
        axis.set_xticks([xx.min(), xx.max()])
        axis.set_yticks([yy.min(), yy.max()])

        if plot_observed_events:
            axis.scatter(observed_events[:, 0], observed_events[:, 1], color='darkgrey', s=3, alpha=1.)

        if inducing_points is None:
            axis.set_title(r'N=%d' % len(observed_events))
            if ground_truth:
                axis.set_title(r'Ground Truth')

        if inducing_points is not None:
            # axis.scatter(inducing_points[:, 0], inducing_points[:, 1], color='grey', s=3, alpha=0.3)
            axis.set_title(r'L=%d/N=%d' % (len(inducing_points), len(observed_events)))

        axis.set_xlabel(label)

        return tmp
