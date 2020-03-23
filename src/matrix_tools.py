import numpy as np
from itertools import product

class MatrixTools:
    def __init__(self):
        pass

    @staticmethod
    def cell_coord_list(*args, **kwargs):
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            ys, xs = args[0].shape
            ys, xs = range(ys), range(xs)
            return list(product(ys, xs))

    def xy_maps(self):
        f = lambda a, b: np.repeat(np.arange(a), b).reshape((a, b))
        return f(self.m_shape[0], self.m_shape[1]), f(self.m_shape[1], self.m_shape[0]).T.copy()

    def xy_diffs(self, ym, xm):
        df = lambda m, s, i, a: np.roll(np.minimum(m, m[::-1]+1), i, axis=a)
        return df(ym.copy(), self.m_shape[0], self.y, 0), df(xm.copy().T, self.m_shape[1], self.x, 0).T  # t comes for shift type

