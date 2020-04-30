from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase

import matplotlib
del matplotlib.font_manager.weight_dict['roman']
matplotlib.font_manager._rebuild()
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams["legend.markerscale"] = 2
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = 'black'

fig = plt.figure(figsize=(6, 6), dpi=100)
# fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
# fig, (ax1, ax2, ax3) = plt.subplot((), projection='3d')


def draw(texture, c, l, erase_number=True):
    if erase_number:
        plt.tick_params(labelbottom=False,
                        labelleft=False,
                        labelright=False,
                        labeltop=False)
    ax.set_xlabel('$\\varphi_1 [{\\rm deg}]$')
    ax.set_ylabel('$\\phi [\\rm deg]$')
    ax.set_zlabel('$\\varphi_2 [\\rm deg]$')
    ax.xaxis._axinfo['juggled'] = (2, 0, 1)
    ax.yaxis._axinfo['juggled'] = (2, 1, 0)
    ax.zaxis._axinfo['juggled'] = (2, 2, 2)

    ax.set_xlim(0, 90)
    ax.set_ylim(0, 90)
    ax.invert_yaxis()
    ax.set_zlim(0, 90)
    ax.invert_zaxis()
    aff = np.diag([1, 1, 1, 1])  # Positioning
    aff[0][3] = -30
    aff[1][3] = 60
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), aff)
    lim_tex = texture[np.where((texture[:, 0] < 100.) & (
        texture[:, 1] < 100.) & (texture[:, 2] < 100.))]
    ax.scatter(lim_tex[:, 0], lim_tex[:, 1], lim_tex[:, 2],
               color=c, marker='o', label=l)
    plt.yticks(range(0, 90 + 1, 30))
    plt.xticks(range(0, 90 + 1, 30))
    ax.set_zticks(range(0, 90 + 1, 30))


def draw_all(texture, erase_number=True):
    if erase_number:
        plt.tick_params(labelbottom=False,
                        labelleft=False,
                        labelright=False,
                        labeltop=False)
    # ax.set_xlabel('$\\varphi_1 [{\\rm deg}]$')
    # ax.set_ylabel('$\\phi [\\rm deg]$')
    # ax.set_zlabel('$\\varphi_2 [\\rm deg]$')
    ax.xaxis._axinfo['juggled'] = (2, 0, 1)
    ax.yaxis._axinfo['juggled'] = (2, 1, 0)
    ax.zaxis._axinfo['juggled'] = (2, 2, 2)

    ax.set_xlim(0, 360)
    ax.set_ylim(0, 180)
    ax.invert_yaxis()
    ax.set_zlim(0, 360)
    ax.invert_zaxis()
    aff = np.diag([1, 0.5, 1, 1])  # Positioning
    aff[0][3] = 100  # x
    aff[1][3] = 100  # y
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), aff)
    ax.plot(texture[:, 0], texture[:, 1], texture[:, 2],
            "o", color="blue", ms=2, mew=0.5)
    plt.yticks(range(0, 180 + 1, 90))
    plt.xticks(range(0, 360 + 1, 90))
    ax.set_zticks(range(0, 360 + 1, 90))


def draw_voxels(voxel_data, use_alpha=False, erase_number=True):
    if erase_number:
        plt.tick_params(labelbottom=False,
                        labelleft=False,
                        labelright=False,
                        labeltop=False)
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 16)
    ax.set_zlim(0, 32)
    ax.xaxis._axinfo['juggled'] = (2, 0, 1)
    ax.yaxis._axinfo['juggled'] = (2, 1, 0)
    ax.zaxis._axinfo['juggled'] = (2, 2, 2)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.invert_zaxis()
    aff = np.diag([0.5, 0.25, 1, 1])  # Positioning
    aff[0][3] = 0  # x
    aff[1][3] = 10  # y
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), aff)
    voxels = np.ceil(voxel_data[:, :, :, 0]).astype(np.bool)
    colors = np.empty((32, 16, 32, 4), dtype=np.float32)
    if use_alpha:
        colors[:, :, :, 0] = 0.
        colors[:, :, :, 1] = 0.
        colors[:, :, :, 2] = 0.
        colors[:, :, :, 3] = voxel_data[:, :, :, 0]
    else:
        colors[:, :, :, :] = cm.jet(voxel_data[:, :, :, 0])
    ax.voxels(voxels, facecolors=colors)

    ax_cb = fig.add_axes([0.9, 0.2, 0.025, 0.5])
    norm = Normalize(vmin=0., vmax=1.)
    cmap = 'binary' if use_alpha else 'jet'
    cbar = ColorbarBase(ax_cb, cmap=cmap, norm=norm, orientation='vertical')
    # cbar.set_ticks(np.unique(voxtex[:, :, :, 0]))
    cbar.set_clim(vmin=0., vmax=1.)
    cbar.solids.set(alpha=1)


def Show():
    plt.show()
