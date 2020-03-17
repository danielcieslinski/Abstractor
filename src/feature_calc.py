import numpy as np
from methodtools import lru_cache
from learn import FeatureMatrix

"""
https://rszalski.github.io/magicmethods/
https://docs.python.org/2/library/itertools.html#itertools.combinations_with_replacement
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_table_of_contents_feature2d/py_table_of_contents_feature2d.html
"""


class FeatureCalculator:
    def __init__(self, y, x, v, m):
        self.y = y
        self.x = x
        self.v = v
        self.m = m

        self.m_shape = np.shape(m)

    # @lru_cache()
    def xy_maps(self):
        f = lambda a, b: np.repeat(np.arange(a), b).reshape((a, b))
        return f(self.m_shape[0], self.m_shape[1]), f(self.m_shape[1], self.m_shape[0]).T.copy()

    def xy_diffs(self, ym, xm):
        df = lambda m, s, i, a: np.roll(np.minimum(m, m[::-1]), i, axis=a)
        return df(ym.copy(), self.m_shape[0], self.y, 0), df(xm.copy(), self.m_shape[1], self.x, 1)  # t comes for shift type

    def matrixer(self):
        return np.full(self.m_shape,self.v)

    def features(self):
        ym, xm = self.xy_maps()
        ydf, xdf = self.xy_diffs(ym, xm)
        self.test()
        # functions = [min, max, avg ==, -, +, *, /, !=, <, >]
        # mt = self.matrixer()
        ma = self.m.m
        v = self.v

        features = np.array((
            ym - self.y,
            xm - self.x,
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

        # print('YM \n', ym)
        # print('XM \n', xm)
        # print('ydf \n', ydf)
        # print('xdf \n', xdf)

        assert all(ym[0, :]) == 0
        assert all(xm[:, 0]) == 0
        assert all(ydf[self.y, :]) == 0
        assert all(xdf[:, self.x]) == 0

# fm = FeatureMatrix.from_raw_values([[1,2,3],[2,3,4],[5,4,3]])
fm = FeatureMatrix.from_raw_values(np.ones((1,6)))
cy, cx = 0, 1
fc = FeatureCalculator(cy, cx, fm[cy,cx], fm)
fc.test()
# print(fc.m)

"""
rules: 
"""
