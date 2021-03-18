# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy import interpolate
from enum import IntEnum


class Datatype(IntEnum):
    train = 0
    valid = 1
    test = 2

# Specify the directory where the training data will be saved.
# Absolute path recommended.
ROOT_DIR = 'E:/'

testdata_dir = ROOT_DIR + 'trainingdata/test/'
traindata_dir = ROOT_DIR + 'trainingdata/train/'
evaldata_dir = ROOT_DIR + 'trainingdata/eval/'

test_listdir = list()
train_listdir = list()
eval_listdir = list()
try:
    train_listdir = os.listdir(traindata_dir + 'texture/')
    eval_listdir = os.listdir(evaldata_dir + 'texture/')
    test_listdir = os.listdir(testdata_dir + 'texture/')
except WindowsError as e:
    print(e)

sscurve_dir = ROOT_DIR + 'trainingdata/data/sscurve/'
rvalue_dir = ROOT_DIR + 'trainingdata/data/rvalue/'
voxel_dir = ROOT_DIR + 'trainingdata/data/voxel/'  # input data dir
image_dir = ROOT_DIR + 'trainingdata/data/image/'

angles = {'0': 0., '45': 0.5, '90': 1.}
order_n = {'0': 0.9, '45': 0.3, '90': 0.1}
plastic_strain = 0.04

ratios = {'1_0': 0., '4_1': 0.125, '2_1': 0.250, '4_3': 0.375,
          '1_1': 0.5, '3_4': 0.625, '1_2': 0.75, '1_4': 0.875, '0_1': 1}

max_vector = np.empty((2, 2))
max_r = np.empty(3)
try:
    max_vector = np.genfromtxt(sscurve_dir + 'max_vector.csv', delimiter=",")
except IOError as e:
    print(e)
try:
    ind_max_r = np.genfromtxt(rvalue_dir + 'max_r.csv', delimiter=",")
except IOError as e:
    print(e)


def func(x, y, tx, method='linear'):
    try:
        return interpolate.interp1d(x, y, kind=method)(tx)
    except ValueError:
        return 0


def get_sscurve(ratio, tex_info, data_type):
    """get raw ss curve data
    Arguments:
        ratio {[type]} -- [description]
        tex_info {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    data_dir = traindata_dir
    if data_type == Datatype.valid:
        data_dir = evaldata_dir
    elif data_type == Datatype.test:
        data_dir = testdata_dir
    data = np.genfromtxt(
        data_dir + ratio + '/' + tex_info + '.dat', delimiter='')
    epsilon_rd = data[:, 13]  # 13 : Logarithmic plastic strain (RD)
    sigama_rd = data[:, 14]  # 14 : true stress (RD)
    epsilon_td = data[:, 15]  # 15 : Logarithmic plastic strain (TD)
    sigama_td = data[:, 16]  # 16 : true stress (TD)
    e = np.array([epsilon_rd, epsilon_td]).T
    s = np.array([sigama_rd, sigama_td]).T
    return e, s


def normalized_sscurve(ratio, tex_info, data_type):
    """ load data from numerical biaxial tensile test

    Args:
        ratio (srting): ex) 1_1
        tex_info (string): ex) 0_02011_01005_02007_01003_03010
        data_type (DataType): ex) DataType.train

    Returns:
        array_like : normalized ss curve
    """
    e, s = get_sscurve(ratio, tex_info, data_type)
    x1 = e[:, 0]
    y1 = s[:, 0]
    x2 = e[:, 1]
    y2 = s[:, 1]
    max_x1 = x1[np.argmax(np.abs(x1))]
    max_y1 = y1[np.argmax(np.abs(y1))]
    max_x2 = x2[np.argmax(np.abs(x2))]
    max_y2 = y2[np.argmax(np.abs(y2))]
    max_values = np.array([[max_x1, max_y1],
                           [max_x2, max_y2]])
    x1 = (x1 / max_values[0, 0])
    y1 = (y1 / max_values[0, 1])
    x2 = (x2 / max_values[1, 0])
    y2 = (y2 / max_values[1, 1])

    # Get 50 points from ss-curve predicted
    # by numerical biaxial tensile test using Linear interpolation
    stress = np.linspace(0.5, 1, 50)
    strain_rd = func(y1, x1, stress)
    strain_td = func(y2, x2, stress)

    # When the stress ratio is 1_0, td is not necessary.
    # The same is true for 0_1
    if ratio == '1_0':
        strain_td = np.arange(0, 50, 1) / 50.
    if ratio == '0_1':
        strain_rd = np.arange(0, 50, 1) / 50.
    strain_teacher = np.array([strain_rd, strain_td])
    return strain_teacher.T, max_values.T


def get_rvalue(angle, tex_info, data_type):
    dataPath = traindata_dir
    if data_type == Datatype.valid:
        dataPath = evaldata_dir
    elif data_type == Datatype.test:
        dataPath = testdata_dir
    # '1_0' means 0 degree, '0_1' means 90 degree
    tmp = angle
    if tmp == '90':
        tmp = '0_1'
    elif tmp == '0':
        tmp = '1_0'
    try:
        data = np.genfromtxt(dataPath + tmp + '/' + tex_info + '.dat', delimiter='')
    except IOError as e:
        print(e)
        return [0], [0]
    x0 = data[:, 13]
    x1 = data[:, 15]
    # When the angle is 90 degrees (ratio is 0_1), swap rd and td
    if angle == '90':
        x0 = data[:, 15]
        x1 = data[:, 13]
    x2 = data[:, 17]  # Logarithmic plastic strain (ND)
    r = x1 / x2
    return x0, r


def r_teacher(angle, tex_info, data_type):
    x0, r = get_rvalue(angle, tex_info, data_type)
    rval = func(x0, r, plastic_strain)
    return rval
