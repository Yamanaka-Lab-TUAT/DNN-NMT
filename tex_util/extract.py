import numpy as np


def random(texture, num):
    if num == 0 or texture.shape[0] == 0:
        return np.empty((0, 3))
    idx = np.random.choice(texture.shape[0], num, replace=False)
    return texture[idx]


def stat(texture, num):
    pass


def hybrid(texture, num):
    pass


method = {'random': random, 'STAT': stat, 'HybridIA': hybrid}
