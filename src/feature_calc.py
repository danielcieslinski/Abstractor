import numpy as np
from methodtools import lru_cache
# from learn import FeatureMatrix

"""
https://rszalski.github.io/magicmethods/
https://docs.python.org/2/library/itertools.html#itertools.combinations_with_replacement
"""


class FeatureCalculator:
    def __init__(self, y, x, v, m):
        self.x = x
        self.y = y
        self.v = v
        self.m = m

        self.m_shape = np.shape(m)

    @lru_cache()
    def xy_maps(self):
        f = lambda a, b: np.repeat(np.arange(a), b).reshape((a, b))
        return f(self.m_shape[0], self.m_shape[1]), f(self.m_shape[1], self.m_shape[0]).T.copy()

    def xy_diffs(self, ym, xm):
        # TODO axis why
        df = lambda m, s, i, a: np.roll(np.abs(np.roll(m - s // 2, - s // 2, axis=a)), i, axis=a)
        return df(ym, self.m_shape[0], self.y, 0), df(xm, self.m_shape[1], self.x, 1)  # t comes for shift type

    def matrixer(self):
        return np.full(self.m_shape,self.v)

    def features(self):
        ydf, xdf = self.xy_diffs(*self.xy_maps())
        # functions = [min, max, avg ==, -, +, *, /, !=, <, >]
        # mt = self.matrixer()
        ma = self.m.m
        v = self.v

        features = np.array((
            ydf,
            xdf,
            ma == v,
            ma < v,
            ma > v,
            ma * v,
            ma - v,
            np.abs(ma - v),
            ma + v
        ))

        return features

        # TODO include counter
        # Count

    # def logical(self, a, b):
    #     return [a and b, a or b, ]

    def relations(self):
        """
        y ~ y | (max_y == y0 + 1 <--> y0 = max_y -1)
        x ~ max_x
        y ~ x
        y ~ y
        x ~ y
        x ~ x
        y ~ v
        x ~ v
        """
        pass

    def combine(self):
        """
        features = [f1, f2, ... fn]
        combinations = [f1f1, f1f2,...f1n, f21, ... fnfn]
        functions = [min, max, avg ==, -, +, *, /, !=, <, >]
        # If only beetwen to erease min and max bcs is equal to < and >


        # Calculate all operations over combinations
        combined = [map(combinations, fun) for fun in functions]

        then calculate logical functions with features
        logical = (and, or, xor)


        rules = [map(features, logi) for in logical]

        append rules to features:
        features += rules

        If correlattion discovered


        """
        pass


    def test(self):
        ym, xm = self.xy_maps()
        ydf, xdf = self.xy_diffs(ym, xm)
        assert all(ym[0, :]) == 0
        assert all(xm[:, 0]) == 0
        assert all(ydf[self.y, :]) == 0
        assert all(xdf[self.x, :]) == 0

# fm = FeatureMatrix.from_raw_values([[1,2,3],[2,3,4],[5,4,3]])
# fc = FeatureCalculator(2, 2, fm[2,2], fm)
# print(fc.m)
