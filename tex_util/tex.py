# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
from tex_util import conv
from tex_util import extract
from tex_util.sym import equivalent

preferred_orientation = {
    'S': np.concatenate((equivalent([59, 37, 63], False),
                         equivalent([53, 75, 34], False),
                         equivalent([27, 58, 18], False))),
    'Goss': equivalent([90, 90, 45], False),
    'Brass': np.concatenate((equivalent([35, 45, 90], False),
                             equivalent([55, 90, 45], False))),
    'Copper': np.concatenate((equivalent([90, 35, 45], False),
                              equivalent([39, 66, 27], False)))
}


def parse(tex_info):
    cube_v = int(tex_info[2:5])
    cube_s = int(tex_info[5:7])
    s_v = int(tex_info[8:11])
    s_s = int(tex_info[11:13])
    goss_v = int(tex_info[14:17])
    goss_s = int(tex_info[17:19])
    brass_v = int(tex_info[20:23])
    brass_s = int(tex_info[23:25])
    copper_v = int(tex_info[26:29])
    copper_s = int(tex_info[29:31])
    rand_vol = 100 - (cube_v + s_v + goss_v + brass_v + copper_v)
    return np.array([cube_v, s_v, goss_v, brass_v, copper_v, rand_vol]), np.array([cube_s, s_s, goss_s, brass_s, copper_s])


def gauss():
    d = np.random.multivariate_normal(
        [0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 100)
    spl = np.array([-1.2815514e+00,
                    -8.41621007e-01,
                    -5.24401007e-01,
                    -2.53347007e-01,
                    0,
                    2.53346992e-01,
                    5.24400992e-01,
                    8.41620992e-01,
                    1.2815514e+00])
    for j in [0, 1, 2]:
        r = d[:, j]
        N = np.empty(10)
        for i in range(10):
            if i == 0:
                N[i] = len(np.where(r <= spl[0])[0])
            elif i == 9:
                N[i] = len(np.where(r > spl[8])[0])
            else:
                a = (np.where(r > spl[i - 1])[0])
                N[i] = len(np.where(r[a] <= spl[i])[0])
        n = np.sum((len(r) / 10 - N) ** 2 / (len(r) / 10))
        if n > 14.7:
            gauss()
    return d


class orientation(object):
    @staticmethod
    def unit(ss, c):
        s0 = 2. / 3.
        vol = 100
        R = np.zeros([3, 3])
        if c == 0:
            R = np.array([[-1.414, 0, 1.414],
                          [0, 2, 0],
                          [-1.414, 0, -1.414]]) / 2
        else:
            R = np.array([[1.414, 0, 1.414],
                          [0, 2, 0],
                          [-1.414, 0, 1.414]]) / 2
        texture = np.zeros([int(vol * s0), 3])
        # note: np.random.rand() generates 0 to 1 random number
        texture[:, 0] = np.random.rand(int(vol * s0)) * 90 * 1.414
        gauss3d = gauss()
        texture[:, 1:] = gauss3d[:int(vol * s0), :2] * \
            np.sqrt(ss)  # 67% : line
        texture = np.dot(texture, R) + np.array([90., 0., 0.])
        texture[texture[:, 1] < 0., 1] *= -1.
        texture[texture[:, 0] < 0., 0] += 90.
        texture[texture[:, 0] > 90., 0] -= 90.
        texture[texture[:, 2] < 0., 2] += 90.
        texture[texture[:, 2] > 90., 2] -= 90.

        texture0 = gauss3d[int(vol * s0):] * np.sqrt(ss)  # 33% : dot
        texture0[texture0[:, 1] > 0, 1] *= -1.
        texture0[(texture0[:, 0] < 0.) & (texture0[:, 2] > 0.)
                 ] += np.array([90., 0., 0.])
        texture0[(texture0[:, 0] < 0.) & (texture0[:, 2] < 0.)
                 ] += np.array([90., 0., 90.])
        texture0[texture0[:, 2] < 0] += np.array([0, 0, 90])
        texture0 += np.array([0, 90, 0])
        return np.concatenate((texture, texture0))

    @staticmethod
    def generate_pseudoTex(ss, vol, ori):
        """Method giving a three-dimensional Gaussian distribution to preferred orientation
        Arguments:
            ss {[type]} -- Dispersion angle
            vol {int} -- Volume fraction
            ori {array_like} -- Preferred orientation considering symmetry
        Returns:
            [array_like] -- preferred orientation with a three-dimensional Gaussian distribution
        """
        if int(vol + 0.5) == 0:
            return np.empty((0, 3))
        tmp = np.empty((0, 3))
        for i in range(ori.shape[0]):
            buf = gauss() * np.sqrt(ss) + ori[i]
            tmp = np.concatenate((tmp, buf))
        return extract.method['random'](tmp, int(vol + 0.5))

    @staticmethod
    def random(vol):
        return np.reshape(np.random.rand(int(vol + 0.5) * 3),
                          (int(vol + 0.5), 3)) * [360, 180, 360]

    @classmethod
    def cube(cls, ss, vol):
        ori = np.empty((0, 3))
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    buf = cls.unit(ss, k)
                    if k == 0:
                        ori = np.concatenate((ori,
                                              buf + np.array([i * 90, 0, j * 90])))
                    else:
                        buf[:, 1] *= -1
                        buf += np.array([i * 90, 180, j * 90])
                        ori = np.concatenate((ori, buf))
        return extract.method['random'](ori, int(vol + 0.5))

    @classmethod
    def goss(cls, ss, vol):
        G = preferred_orientation['Goss']
        return cls.generate_pseudoTex(ss, int(vol + 0.5), G)

    @classmethod
    def brass(cls, ss, vol):
        B = preferred_orientation['Brass']
        return cls.generate_pseudoTex(ss, int(vol + 0.5), B)

    @classmethod
    def copper(cls, ss, vol):
        Cu = preferred_orientation['Copper']
        return cls.generate_pseudoTex(ss, int(vol + 0.5), Cu)

    @classmethod
    def S(cls, ss, vol):
        S = preferred_orientation['S']
        return cls.generate_pseudoTex(ss, int(vol + 0.5), S)


class Texture(object):
    def __init__(self, volume=1000, texture=np.empty((0, 3))):
        self.tex_data = texture
        self.vol = volume
        self.vol_max = volume

    def __clamp(self, val, max_val=255, min_val=0):
        val[val > max_val] = max_val
        val[val < min_val] = min_val
        return val

    def add(self, texture):
        self.tex_data = np.concatenate((self.tex_data, texture))

    def addRandom(self, percentage):
        self.add(orientation.random(self.vol_max * percentage / 100.))

    def addCube(self, ss, percentage):
        self.add(orientation.cube(ss, self.vol_max * percentage / 100.))

    def addS(self, ss, percentage):
        self.add(orientation.S(ss, self.vol_max * percentage / 100.))

    def addGoss(self, ss, percentage):
        self.add(orientation.goss(ss, self.vol_max * percentage / 100.))

    def addBrass(self, ss, percentage):
        self.add(orientation.brass(ss, self.vol_max * percentage / 100.))

    def addCopper(self, ss, percentage):
        self.add(orientation.copper(ss, self.vol_max * percentage / 100.))

    def pole_figure(self, direct=np.array([1, 1, 1]), invert=False,
                    denominator=10, img_size=128, method='random'):
        """Positive pole figure
        Keyword Arguments:
            direct {array_like} -- Direction of projection plane (default: {np.array([1, 1, 1])})
            denominator {int} -- N (default: {10})
            img_size {int} -- image size (default: {128})
            method {str} -- Extraction method (default: {'random'})
        Returns:
            numpy array (img_size, img_size) -- pole figure ss image data
        """
        stereo = conv.stereo(self.sample(method), direct)
        stereo = np.array([stereo / 2. * img_size / 2. + img_size / 2.], dtype="uint8")
        img = np.zeros([img_size, img_size])
        for i, j in stereo[0]:
            img[i, j] += 1
        img = img / denominator * 255.0 if denominator != 0 else np.ceil(img) * 255
        self.__clamp(img)
        if invert:
            img = np.ones([img_size, img_size]) * 255 - img
        return img

    def sample(self, method='random'):
        """Extract crystal orientation
        Keyword Arguments:
            method {str} -- Extract method (default: {'random'})
        Returns:
            [numpy array] -- [description]
        """
        tmp = extract.method[method](self.tex_data, self.vol)
        tmp[tmp < 0.] += 360.
        tmp[tmp > 360.] -= 360.
        return tmp

    def voxel(self, div=32, denominator=40, method='random'):
        temp = self.sample(method)
        temp /= 360.
        # temp[:, 1] *= 2.  # ??
        temp *= div - 1.
        point = np.array(np.floor(temp + 0.5), dtype='uint8')
        vox = np.zeros([div, int(div / 2), div, 1], dtype='float16')
        for (phi1, phi2, phi3) in point:
            vox[phi1, phi2, phi3, 0] += 1.
        vox /= float(denominator)
        self.__clamp(vox, max_val=1.)
        return vox

    def saveTxt(self, file_name, method='random'):
        data_to_save = []
        append = data_to_save.append
        cnt = 1
        for euler in self.sample(method):
            tmp = [cnt, 1, euler[0], euler[1], euler[2]]
            append(tmp)
            cnt += 1
        np.savetxt(file_name, np.array(data_to_save), delimiter='\t',
                   fmt=['%4d', '%1d', '% 3.5f', '% 3.5f', '% 3.5f'])

    def savePoleFigure(self, file_name, direct=np.array([1, 1, 1]), invert=False,
                       denominator=10, img_size=128, method='random'):
        pil = Image.fromarray(np.uint8(self.pole_figure(
            direct, invert, denominator, img_size, method)))
        pil.save(file_name)

    def fromtxt(self, tex_info_path):
        data = np.genfromtxt(tex_info_path, delimiter='')
        phi1 = data[:, 2]
        phi = data[:, 3]
        phi2 = data[:, 4]
        self.tex_data = np.array([phi1, phi, phi2]).T
        return self


if __name__ == '__main__':
    if False:
        tex = Texture(volume=1000)
        B_V = 22
        B_S = np.random.randint(8, 15)
        tex.addBrass(B_S, B_V)
        G_V = 4
        G_S = np.random.randint(8, 15)
        tex.addGoss(G_S, G_V)
        S_V = 41
        S_S = np.random.randint(8, 15)
        tex.addS(S_S, S_V)
        C_V = 4
        C_S = np.random.randint(8, 15)
        tex.addCopper(C_S, C_V)
        Cube_V = 5
        Cube_S = np.random.randint(8, 15)
        tex.addCube(Cube_S, Cube_V)

        random_v = 100 - Cube_V - S_V - G_V - B_V - C_V
        tex.addRandom(random_v)
        print(random_v)

        filename = '0_' + '{0:03d}'.format(int(Cube_V))
        filename += '{0:02d}'.format(int(Cube_S)) + '_'
        filename += '{0:03d}'.format(int(S_V))
        filename += '{0:02d}'.format(int(S_S)) + '_'
        filename += '{0:03d}'.format(int(G_V))
        filename += '{0:02d}'.format(int(G_S)) + '_'
        filename += '{0:03d}'.format(int(B_V))
        filename += '{0:02d}'.format(int(B_S)) + '_'
        filename += '{0:03d}'.format(int(C_V))
        filename += '{0:02d}'.format(int(C_S)) + '.txt'
        tex.saveTxt(filename)
        tex.savePoleFigure('CUBE.png')

    import myplt
    tex = Texture(volume=1000)
    tex.addGoss(10, 100)
    myplt.draw_voxels(tex.voxel(), use_alpha=False)
    myplt.Show()
    # tex.savePoleFigure('S.png', denominator=4)
    # myplt.draw(tex.sample(), 'b', 'Brass orientation')
    # myplt.draw_all(tex.sample())
    # handles, labels = myplt.ax.get_legend_handles_labels()
    # myplt.fig.legend(handles, labels, loc='center left', borderaxespad=0)
