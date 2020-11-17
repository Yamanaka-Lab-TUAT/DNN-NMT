# -*- coding: utf-8 -*-
import nnabla as nn
import numpy as np
import rawdata as rdat
from tex_util import Texture

""" from nnc_proj.($$project_name_of_nnc) import network """
from nnc_proj.model import network

nn.clear_parameters()
nn.parameter.load_parameters('./nnc_proj/model.nnp')

x = nn.Variable((1, 1, 128, 128))
x2 = nn.Variable((1, 1))
y1, y2 = network(x, x2, test=True)

ratios = rdat.ratios
mvec = rdat.max_vector


def estimate_sscurve(stress_roots, tex_info, num=1):
    '''
    return:(array_like, [rootNum, 50]) strain, stress
    '''
    x = nn.Variable((num, 1, 128, 128))
    x2 = nn.Variable((num, 1))
    y1, y2 = network(x, x2, test=True)
    # RD, TD なので2つ [num, xx, '2']
    rootsNum = len(stress_roots)
    strain = np.empty((num, rootsNum, 50, 2))
    stress = np.empty((num, rootsNum, 50, 2))
    y_norm = np.array([np.linspace(0.5, 1, 50) for j in range(2)]).T
    for cnt in range(num):
        tex = Texture(volume=1000, tex_info=tex_info)
        img = tex.pole_figure()
        x.d[cnt, 0] = img / 255.0
    for i, root in enumerate(stress_roots):
        x2.d = ratios[root]
        y1.forward()
        y2.forward()
        curve = y1.d[:]
        max_vec = y2.d[:] * mvec
        for cnt in range(num):
            strain[cnt, i] = curve[cnt] * max_vec[cnt, 0, :]
            stress[cnt, i] = y_norm * max_vec[cnt, 1, :]
    ave_stress = np.average(stress, axis=0)
    ave_strain = np.average(strain, axis=0)
    std = np.std(stress, axis=0)

    # 配列の先頭に0を追加
    ave_stress = np.insert(ave_stress, 0, 0, axis=1)
    ave_strain = np.insert(ave_strain, 0, 0, axis=1)
    std = np.insert(std, 0, 0, axis=1)

    return ave_strain, ave_stress, std


if __name__ == "__main__":
    # texture = '0_00514_04109_00406_02209_01611'
    # texture = '0_00611_02907_00507_01507_01311'
    texture = '0_01813_00808_01012_00213_00405'

    roots = ['1_0', '4_1', '2_1', '4_3', '1_1', '3_4', '1_2', '1_4', '0_1']
    import time
    start = time.time()
    strain, stress, std = estimate_sscurve(roots, texture, 50)
    elapsed = time.time() - start
    print('elapsed time: {} [sec]'.format(elapsed))
    np.save(strain, 'estimate_result/2dcnn_strain_{}.npy'.format(texture))
    np.save(stress, 'estimate_result/2dcnn_stress_{}.npy'.format(texture))
    # np.save(std, 'estimate_result/2dcnn_strain_{}.npy'.format(texture))
