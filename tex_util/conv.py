# -*- coding: utf-8 -*-
import numpy as np


def rodrigues(miller_dict, theta, rounding=True):
    '''rodrigues matrix

    Arguments:
        miller_dict {list or np.array} -- axis of rotation
        theta {float} -- angle [deg]
        rounding {Bool} -- do or not rounding

    Returns:
        np.array3x3 -- rodrigues mat
    '''
    norm = np.linalg.norm(miller_dict)
    miller_dict = miller_dict / norm if norm > 1e-5 else miller_dict
    cos = np.cos(np.deg2rad(theta))
    sin = np.sin(np.deg2rad(theta))
    u = miller_dict[0]
    v = miller_dict[1]
    w = miller_dict[2]
    g = np.array([[cos + u**2. * (1 - cos), u * v * (1 - cos) - w * sin, u * w * (1 - cos) + v * sin],
                  [u * v * (1 - cos) + w * sin, cos + v**2. * (1 - cos), v * w * (1 - cos) - u * sin],
                  [u * w * (1 - cos) - v * sin, v * w * (1 - cos) + u * sin, cos + w**2. * (1 - cos)]])
    return np.round(g) if rounding else g


def rotate_mat(bunge_euler):
    '''R matrix

    Arguments:
        phi {list} -- bunge's euler [phi1, phi, phi2]

    Returns:
        numpy array3x3
    '''
    phi = np.deg2rad(bunge_euler)
    Z1 = np.array([[np.cos(phi[0]), np.sin(phi[0]), 0],
                   [-np.sin(phi[0]), np.cos(phi[0]), 0],
                   [0, 0, 1]])
    X = np.array([[1, 0, 0],
                  [0, np.cos(phi[1]), np.sin(phi[1])],
                  [0, -np.sin(phi[1]), np.cos(phi[1])]])
    Z2 = np.array([[np.cos(phi[2]), np.sin(phi[2]), 0],
                   [-np.sin(phi[2]), np.cos(phi[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Z2, np.dot(X, Z1))
    return R


def euler2pole(phi, direct):
    ''' The southern hemisphere is also inverted and projected '''
    R = rotate_mat(phi)
    direct = direct / np.linalg.norm(direct)
    o = np.dot(R.T, direct.T)
    return o


def stereo(phi_array, direct=np.array([1, 1, 1])):
    '''Convert Bunge euler angle to projected pole figure
        with the specified orientation.

    Arguments:
        phi_array {array_like} -- Bunge euler angle

    Keyword Arguments:
        direct {array_like} -- Direction of plane to project (default: {np.array([1, 1, 1])})

    Returns:
        numpy array (2L, :) -- Coordinates on 2D plane
    '''
    projection_point = []
    for phi in phi_array:
        pole = euler2pole(phi, direct)
        # if pole[2] >= 0.:
        #     projection_point.append([2 * pole[0] / (1 + pole[2]),
        #                             2 * pole[1] / (1 + pole[2])])
        projection_point.append([2 * pole[0] / (1 + np.abs(pole[2])),
                                 2 * pole[1] / (1 + np.abs(pole[2]))])
    return np.array(projection_point)


def miller2euler(h, k, l, u, v, w, rounding=True):
    """convert miller indicate to bunge's euler angle

    Arguments:
        h {int} -- (h k l)[u v w]
        k {int} -- (h k l)[u v w]
        l {int} -- (h k l)[u v w]
        u {int} -- (h k l)[u v w]
        v {int} -- (h k l)[u v w]
        w {int} -- (h k l)[u v w]
    """
    # print(f"\n({h} {k} {l})[{u} {v} {w}]")
    # print("({0} {1} {2})[{3} {4} {5}]".format(h, k, l, u, v, w))
    delta1 = np.sqrt(u**2. + v**2. + w**2.)
    delta2 = np.sqrt(h**2. + k**2. + l**2.)
    delta3 = np.sqrt(h**2. + k**2.)
    sin_f1 = w * delta2 / (delta1 * delta3)
    cos_f = l / delta2
    cos_f2 = k / delta3
    ret = np.array([np.arcsin(sin_f1), np.arccos(cos_f), np.arccos(cos_f2)]) * 180. / np.pi
    return np.round(ret) if rounding else ret


if __name__ == "__main__":
    # print(miller2euler(-2, -3, -1, 3, -4, 6))
    # print(miller2euler(1, 3, 2, 6, -4, 3))
    # print(miller2euler(2, 1, 3, -3, -6, 4))
    # print(miller2euler(2, 3, 1, 3, -4, 6))
    # print(miller2euler(-1, -2, -3, 6, 3, -4))
    print(miller2euler(1, 0, 1, 1, 2, 2, False))
    print(miller2euler(0, 1, 1, 1, 2, 2, False))
    print(miller2euler(0, 1, 1, 2, -2, 1, False))
