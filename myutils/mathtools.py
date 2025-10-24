import itertools as itt
import numpy as np
from scipy.stats import entropy


def hv2d(locations, ref=None):
    if ref is None:
        ref = (0.0, 0.0)
    n = len(locations)
    locations.sort(key=lambda p:p[0])
    cleaned = [*range(n)]
    for i in range(n):
        x1, y1 = locations[i]
        for j in range(n):
            if j == i:
                continue
            x2, y2 = locations[j]
            if x1 <= x2 and y1 <= y2 and (x1 < x2 or y1 < y2):
                cleaned.remove(i)
                break

    rx, ry = ref
    hv = 0
    # last_
    x_prev = rx
    for i in cleaned:
        x, y = locations[i]
        if x <= rx or y <= ry:
            continue
        hv += (x - x_prev) * (y - ry)
        x_prev = x
    return hv

def a_clip(v, g, r=1.0):
    return min(r, 1 - abs(v - g) / g)

def jsdiv(p, q):
    return (entropy(p, p + q, base=2) + entropy(q, p + q, base=2)) / 2

def box_to_euc(x):
    r = np.abs(x).max() / np.linalg.vector_norm(x)
    return x * r

def euc_clip(x, maximum):
    m = np.linalg.vector_norm(x)
    r = 1.0 if m <= maximum else maximum / m
    return x * r

def euc_clip_box(x, maximum):
    return euc_clip(box_to_euc(x), maximum)

if __name__ == '__main__':
    # print(box_to_euc(np.array([1., 1., 1., 1.])))
    # print(euc_clip_box(np.array([2., 2., 2., 2.]), 1.0))
    # print(euc_clip_box(np.array([1., 1., 1., 1.]), 1.0))
    # print(euc_clip_box(np.array([.1, .1, .1, .1]), 1.0))
    # print(euc_clip_box(np.array([2., 0., 0., 0.]), 1.0))
    # print(euc_clip_box(np.array([1., 0., 0., 0.]), 1.0))
    # print(euc_clip_box(np.array([.1, 0., 0., 0.]), 1.0))
    print(hv2d([(10, 9), (5, 15), (3, 12), (8, 13)]))
