import numpy as np
from matplotlib import pyplot as plt
import common.graph_setup as graph 
import common.rawdata as rdat
import dnn2d


if __name__ == '__main__':
    fig, axes = graph.ss_curve_setup()

    # texture = '0_00514_04109_00406_02209_01611'
    # texture = '0_00611_02907_00507_01507_01311'
    texture = '0_01813_00808_01012_00213_00405'

    roots = ['1_0', '4_1', '2_1', '4_3', '1_1', '3_4', '1_2', '1_4', '0_1']
    import time
    start = time.time()
    strain, stress, std = dnn2d.estimate_sscurve(roots, texture, 5)
    elapsed = time.time() - start
    print('elapsed time: {} [sec]'.format(elapsed))
    cnt = 0
    for axe in axes:
        for ax in axe:
            graph.draw_sscurve(ax, strain[cnt], stress[cnt], std[cnt], roots[cnt])
            cnt += 1
    plt.show()
