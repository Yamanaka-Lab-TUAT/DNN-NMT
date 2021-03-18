# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
import common.rawdata as rdat
from common.rawdata import Datatype
from tex_util import Texture


dat_path = rdat.sscurve_dir

save_dir = './label/'  # Path where to save the label for NNC

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


if not os.path.exists(rdat.image_dir):
    os.makedirs(rdat.image_dir)


def get_texture(tex_info, data_type):
    dataPath = rdat.traindata_dir
    if data_type == Datatype.valid:
        dataPath = rdat.evaldata_dir
    elif data_type == Datatype.test:
        dataPath = rdat.testdata_dir
    data = np.genfromtxt(
        dataPath + 'texture/' + tex_info + '.txt', delimiter='')
    phi1 = data[:, 2]
    phi = data[:, 3]
    phi2 = data[:, 4]
    return np.array([phi1, phi, phi2]).T


def create_image_input():
    for lt in rdat.train_listdir:
        tex_info = lt.rstrip('.txt')
        raw_data = get_texture(tex_info, data_type=0)
        tex_data = Texture(texture=raw_data)
        tex_data.savePoleFigure(rdat.image_dir + 'data_' + tex_info + '.png')
    for lt in rdat.eval_listdir:
        tex_info = lt.rstrip('.txt')
        raw_data = get_texture(tex_info, data_type=1)
        tex_data = Texture(texture=raw_data)
        tex_data.savePoleFigure(rdat.image_dir + 'data_' + tex_info + '.png')


def save_sscurve_dataset(file_name, data):
    with open(file_name, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(["x:input", "r:stressx", "s:curve", "e:vector"])
        writer.writerows(data)


def create_teacher_data():
    # ss curve
    max_vec = rdat.max_vector
    cnt = 0
    for ratio in rdat.ratios:
        for lt in rdat.train_listdir:
            tex_info = lt.rstrip('.txt')
            save_curve_dir = dat_path + ratio + '/curve/'
            save_vector_dir = dat_path + ratio + '/vector/'
            if not os.path.exists(save_curve_dir):
                os.makedirs(save_curve_dir)
            if not os.path.exists(save_vector_dir):
                os.makedirs(save_vector_dir)
            curve, vector = rdat.normalized_sscurve(ratio, tex_info, 0)
            vector = vector / max_vec
            # save ss curve
            np.savetxt(save_curve_dir + 'data_' + tex_info + '.csv',
                       curve, comments='', delimiter=",")
            np.savetxt(save_vector_dir + 'data_' + tex_info + '.csv',
                       vector, comments='', delimiter=",")
            cnt += 1
    print('finish ss curve train')
    # validation data
    cnt = 0
    for ratio in rdat.ratios:
        for lt in rdat.eval_listdir:
            tex_info = lt.rstrip('.txt')
            curve, vector = rdat.normalized_sscurve(ratio, tex_info, 1)
            vector = vector / max_vec
            save_curve_dir = dat_path + ratio + '/curve/'
            save_vector_dir = dat_path + ratio + '/vector/'
            if not os.path.exists(save_curve_dir):
                os.makedirs(save_curve_dir)
            if not os.path.exists(save_vector_dir):
                os.makedirs(save_vector_dir)
            # save ss curve
            np.savetxt(save_curve_dir + 'data_' + tex_info + '.csv',
                       curve, comments='', delimiter=",")
            np.savetxt(save_vector_dir + 'data_' + tex_info + '.csv',
                       vector, comments='', delimiter=",")
            cnt += 1
    print('finish ss curve eval')
    create_image_input()
    print('finish input data')


def createdataset():
    train_sscurve_label = []
    eval_sscurve_label = []
    tr_cnt = 0
    for cnt in range(1):
        for ratio, val in rdat.ratios.items():
            for lt in rdat.train_listdir:
                tr_cnt += 1
                tex_info = lt.rstrip('.txt')
                saveTexName = str(cnt) + '_' + tex_info[2:]
                train_sscurve_label.append(
                    [rdat.image_dir + "data_" + saveTexName + '.png',
                     val,
                     dat_path + "%s/curve/data_" % ratio + tex_info + ".csv",
                     dat_path + "%s/vector/data_" % ratio + tex_info + ".csv"])
    ev_cnt = 0
    for ratio, val in rdat.ratios.items():
        for lt in rdat.eval_listdir:
            ev_cnt += 1
            tex_info = lt.rstrip('.txt')
            eval_sscurve_label.append(
                [rdat.image_dir + "data_" + tex_info + '.png',
                 val,
                 dat_path + "%s/curve/data_" % ratio + tex_info + ".csv",
                 dat_path + "%s/vector/data_" % ratio + tex_info + ".csv"])
    save_sscurve_dataset(save_dir + 'sscurve_train.csv', train_sscurve_label)
    save_sscurve_dataset(save_dir + 'sscurve_eval.csv', eval_sscurve_label)
    print('ss curve Train    : ' + str(tr_cnt) + ' data')
    print('ss curve Evaluate : ' + str(ev_cnt) + ' data')


if __name__ == '__main__':
    create_teacher_data()
    createdataset()
