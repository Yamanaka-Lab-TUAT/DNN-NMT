import numpy as np
# import conv
from tex_util import conv

E = conv.rodrigues(miller_dict=[0, 0, 0], theta=0)

C4x_1 = conv.rodrigues(miller_dict=[1, 0, 0], theta=90)
C4x_2 = conv.rodrigues(miller_dict=[1, 0, 0], theta=180)
C4x_3 = conv.rodrigues(miller_dict=[1, 0, 0], theta=270)
C4y_1 = conv.rodrigues(miller_dict=[0, 1, 0], theta=90)
C4y_2 = conv.rodrigues(miller_dict=[0, 1, 0], theta=180)
C4y_3 = conv.rodrigues(miller_dict=[0, 1, 0], theta=270)
C4z_1 = conv.rodrigues(miller_dict=[0, 0, 1], theta=90)
C4z_2 = conv.rodrigues(miller_dict=[0, 0, 1], theta=180)
C4z_3 = conv.rodrigues(miller_dict=[0, 0, 1], theta=270)

C31_1 = conv.rodrigues(miller_dict=[1, 1, 1], theta=120)
C31_2 = conv.rodrigues(miller_dict=[1, 1, 1], theta=240)
C32_1 = conv.rodrigues(miller_dict=[1, 1, -1], theta=120)
C32_2 = conv.rodrigues(miller_dict=[1, 1, -1], theta=240)
C33_1 = conv.rodrigues(miller_dict=[-1, 1, 1], theta=120)
C33_2 = conv.rodrigues(miller_dict=[-1, 1, 1], theta=240)
C34_1 = conv.rodrigues(miller_dict=[1, -1, 1], theta=120)
C34_2 = conv.rodrigues(miller_dict=[1, -1, 1], theta=240)

C2a = conv.rodrigues(miller_dict=[1, 0, 1], theta=180)
C2b = conv.rodrigues(miller_dict=[-1, 0, 1], theta=180)
C2c = conv.rodrigues(miller_dict=[1, 1, 0], theta=180)
C2d = conv.rodrigues(miller_dict=[-1, 1, 0], theta=180)
C2e = conv.rodrigues(miller_dict=[0, 1, 1], theta=180)
C2f = conv.rodrigues(miller_dict=[0, -1, 1], theta=180)

sym_mat = [E, C4x_1, C4x_2, C4x_3, C4y_1, C4y_2, C4y_3, C4z_1,
           C4z_2, C4z_3, C31_1, C31_2, C32_1, C32_2, C33_1,
           C33_2, C34_1, C34_2, C2a, C2b, C2c, C2d, C2e, C2f]


def myclamp(rad):
    tmp = rad
    if rad < 0:
        tmp += 2 * np.pi
    return tmp


def equivalent(bunge_euler, rounding=True):
    ret = []
    R = conv.rotate_mat(bunge_euler)
    for C in sym_mat:
        R_dash = np.dot(C, R)
        phi = myclamp(np.arccos(R_dash[2, 2])) * 180 / np.pi
        phi1 = myclamp(np.arctan2(R_dash[2, 0], -R_dash[2, 1])) * 180 / np.pi
        phi2 = myclamp(np.arctan2(R_dash[0, 2], R_dash[1, 2])) * 180 / np.pi
        ret.append([phi1, phi, phi2])
    return np.round(np.array(ret)) if rounding else np.array(ret)


if __name__ == '__main__':
    print(equivalent([53., 106., 146.]))
