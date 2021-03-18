import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
# plt.rcParams['axes.grid'] = True
# plt.rcParams['grid.linestyle'] = '-'
# plt.rcParams['legend.markerscale'] = 2
plt.rcParams['legend.fancybox'] = False
plt.rcParams['legend.framealpha'] = 1
plt.rcParams['legend.edgecolor'] = 'black'


def draw_sscurve(ax, x_vec, y_vec, std, ratio, skipping=False, show_std=False):
    if skipping:
        x_vec = x_vec[::2]
        y_vec = y_vec[::2]
        std = std[::2]
    if ratio == '1_0':
        # ax.plot(e[:, 0], s[:, 0], 'k-', linewidth=0.5)
        if not (x_vec is None or y_vec is None):
            if show_std:
                ax.fill_between(
                    x_vec[:, 0], y_vec[:, 0] - std[:, 0],
                    y_vec[:, 0] + std[:, 0], facecolor='r', alpha=0.4)
                ax.plot(
                    x_vec[:, 0], y_vec[:, 0], 'r-', linewidth=0.7,
                    markeredgewidth=0.5, markersize=1.5)
            else:
                ax.plot(
                    x_vec[:, 0], y_vec[:, 0], 'r-', linewidth=0.7,
                    markeredgewidth=0.5, markersize=1.5)
    elif ratio == '0_1':
        # ax.plot(e[:, 1], s[:, 1], 'k--', linewidth=0.5)
        if not (x_vec is None or y_vec is None):
            if show_std:
                ax.fill_between(
                    x_vec[:, 1], y_vec[:, 1] - std[:, 1],
                    y_vec[:, 1] + std[:, 1], facecolor='r', alpha=0.2)
                ax.plot(
                    x_vec[:, 1], y_vec[:, 1], 'r-', linewidth=0.7,
                    markeredgewidth=0.5, markersize=1.5)
            else:
                ax.plot(x_vec[:, 1], y_vec[:, 1], 'r--', linewidth=0.7,
                        markeredgewidth=0.5, markersize=1.5)
    else:
        # ax.plot(e[:, 0], s[:, 0], 'k-', linewidth=0.5)
        # ax.plot(e[:, 1], s[:, 1], 'k--', linewidth=0.5)
        if not (x_vec is None or y_vec is None):
            if show_std:
                ax.fill_between(
                    x_vec[:, 0], y_vec[:, 0] - std[:, 0],
                    y_vec[:, 0] + std[:, 0], facecolor='r', alpha=0.2)
                ax.plot(
                    x_vec[:, 0], y_vec[:, 0], 'r-', linewidth=0.7,
                    markeredgewidth=0.5, markersize=1.5)
                ax.fill_between(
                    x_vec[:, 1], y_vec[:, 1] - std[:, 1],
                    y_vec[:, 1] + std[:, 1], facecolor='r', alpha=0.2)
                ax.plot(
                    x_vec[:, 1], y_vec[:, 1], 'r-', linewidth=0.7,
                    markeredgewidth=0.5, markersize=1.5)
            else:
                ax.plot(x_vec[:, 0], y_vec[:, 0], 'r-', linewidth=0.7,
                        markeredgewidth=0.5, markersize=1.5)
                ax.plot(x_vec[:, 1], y_vec[:, 1], 'r--', linewidth=0.7,
                        markeredgewidth=0.5, markersize=1.5)


def draw_rvalue(graph, angles, r_value, r_std=None, marker='ko'):
    if r_std is None:
        graph.plot(angles, r_value, marker, markersize=4, linewidth=1, clip_on=False)
    else:
        e = graph.errorbar(angles, r_value, yerr=r_std, fmt=marker, capsize=4)
        for b in e[1]:
            b.set_clip_on(False)


def ss_curve_setup():
    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(7, 8), dpi=100)
    cnt = 0
    for axe in axes:
        for ax in axe:
            ax.set_xlim([-0.02, 0.06])
            ax.set_ylim([0, 300])
            ax.set_xticks(np.arange(-0.02, 0.08, 0.02))
            ax.set_xticks(np.arange(-0.02, 0.07, 0.01), minor=True)
            ax.set_yticks(np.arange(0, 300 + 100, 100))
            ax.set_yticks(np.arange(0, 300 + 50, 50), minor=True)
            # ax.set_yticks(np.arange(0, 350 + 50, 50))
            # ax.set_yticks(np.arange(00, 350 + 25, 25), minor=True)
            ax.set_xlabel('Logarithmic plastic strain $\\epsilon^{\\rm p}$')
            ax.set_ylabel('True stress $\\sigma$' + ' [MPa]')
            # ax.grid(which='both', color='#AAAAAA', linestyle='solid')
            cnt += 1
    plt.subplots_adjust(wspace=0.5, hspace=0.4)
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.07, top=0.9)
    axes[0, 0].plot([None], [None], 'k-', linewidth=0.7,
                    markeredgewidth=0.5, markersize=1.5, label='Simulation RD')
    axes[0, 0].plot([None], [None], 'k--', linewidth=0.7,
                    markeredgewidth=0.5, markersize=1.5, label='Simulation TD')
    axes[0, 0].plot([None], [None], 'r-', linewidth=0.7, markeredgewidth=0.5,
                    markersize=1.5, label='Trained DNN RD')
    axes[0, 0].plot([None], [None], 'r--', linewidth=0.7, markeredgewidth=0.5,
                    markersize=1.5, label='Trained DNN TD')
    axes[0, 0].legend(ncol=2, bbox_to_anchor=(0., 1.02, 1., 0.2), 
                      loc='lower left')
    return fig, axes


def rvalue_graph_setup():
    fig, ax = plt.subplots(ncols=1, figsize=(3, 3), dpi=200)
    ax.set_xlim([0, 90])
    ax.set_ylim([0, 2])
    ax.set_xticks(np.linspace(0, 90, 5))
    ax.set_xticks(np.linspace(0, 90, 9), minor=True)
    ax.set_yticks(np.linspace(0, 2, 5))
    ax.set_yticks(np.linspace(0, 2, 9), minor=True)
    ax.set_xlabel('Logarithmic plastic strain $\\epsilon$$^{\\rm{p}}$')
    ax.set_ylabel('$\\it{r}$ - value')
    # ax.grid(which='both', color='#AAAAAA', linestyle='solid')
    plt.subplots_adjust(wspace=0.4)
    plt.subplots_adjust(left=0.15, right=0.9, bottom=0.2, top=0.9)
    return fig, ax
