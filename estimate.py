# -*- coding: utf-8 -*-
import nnabla as nn
import numpy as np
import rawdata as rdat
from tex_util import Texture, parse

""" from nnc_proj.($$project_name_of_nnc) import network """
from nnc_proj.model import network

nn.clear_parameters()
nn.parameter.load_parameters('./nnc_proj/model.nnp')

x = nn.Variable((1, 1, 128, 128))
x2 = nn.Variable((1, 1))
y1, y2 = network(x, x2, test=True)

ratios = rdat.ratios
mvec = rdat.max_vector


def calc(ratio, tex_info, num=1):
    '''
    return:(array_like, [50, 2(RD:0 or TD:1)]) strain, stress
    '''
    vol, frac = parse(tex_info)
    # RD, TD なので2つ [num, xx, '2']
    ret1 = np.empty([num, 50, 2])
    ret2 = np.empty([num, 2, 2])
    for cnt in range(num):
        tex = Texture(volume=1000)
        tex.addCube(frac[0], vol[0])
        tex.addS(frac[1], vol[1])
        tex.addGoss(frac[2], vol[2])
        tex.addBrass(frac[3], vol[3])
        tex.addCopper(frac[4], vol[4])
        tex.addRandom(vol[5])
        img = tex.pole_figure()
        x.d[0, 0] = img / 255.0
        x2.d = ratio
        y1.forward()
        y2.forward()
        ret1[cnt, :, 0] = y1.d[0, :, 0]
        ret2[cnt, :, 0] = y2.d[0, :, 0]
        ret1[cnt, :, 1] = y1.d[0, :, 1]
        ret2[cnt, :, 1] = y2.d[0, :, 1]
    avecurve = np.average(ret1, axis=0)
    avevector = np.average(ret2, axis=0) * mvec
    y = np.array([np.linspace(0.5, 1, 50) for i in range(2)]).T  # RD, TDで２つ
    strain = avecurve * avevector[0]
    stress = y * avevector[1]
    return strain, stress


def calcAllRoots(rootNum, tex_info, num=1):
    '''
    return:(array_like, [rootNum, 50]) strain, stress
    '''
    vol, frac = parse(tex_info)
    # RD, TD なので2つ [num, xx, '2']
    strain = np.empty((num, rootNum, 50, 2))
    stress = np.empty((num, rootNum, 50, 2))
    y_norm = np.array([np.linspace(0.5, 1, 50) for j in range(2)]).T
    for cnt in range(num):
        tex = Texture(volume=1000)
        tex.addCube(frac[0], vol[0])
        tex.addS(frac[1], vol[1])
        tex.addGoss(frac[2], vol[2])
        tex.addBrass(frac[3], vol[3])
        tex.addCopper(frac[4], vol[4])
        tex.addRandom(vol[5])
        # tex.savePoleFigure(tex_info + '.png', invert=True, denominator=0)
        img = tex.pole_figure()
        x.d[0, 0] = img / 255.0
        for root in range(rootNum):
            x2.d = root / float(rootNum - 1)
            y1.forward()
            y2.forward()
            y2.d[0] *= mvec
            curve = y1.d
            max_vec = y2.d
            strain[cnt, root] = curve[0, :] * max_vec[0, 0]
            stress[cnt, root] = y_norm * max_vec[0, 1]
    ave_stress = np.average(stress, axis=0)
    ave_strain = np.average(strain, axis=0)
    std = np.std(stress, axis=0)
    return ave_strain, ave_stress, std


if __name__ == "__main__":
    # texture = '0_00514_04109_00406_02209_01611'
    # texture = '0_00611_02907_00507_01507_01311'
    texture = '0_01813_00808_01012_00213_00405'

    roots = ['1_0', '4_1', '2_1', '4_3', '1_1', '3_4', '1_2', '1_4', '0_1']
    import time
    start = time.time()
    strain, stress, std = calcAllRoots(len(roots), texture, 50)
    elapsed = time.time() - start
    print('elapsed time: {} [sec]'.format(elapsed))
    np.save(strain, 'estimate_result/2dcnn_strain_{}.npy'.format(texture))
    np.save(stress, 'estimate_result/2dcnn_stress_{}.npy'.format(texture))
    # np.save(std, 'estimate_result/2dcnn_strain_{}.npy'.format(texture))
