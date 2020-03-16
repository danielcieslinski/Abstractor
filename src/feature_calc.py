import numpy as np
from methodtools import lru_cache

class FeatureCalculator:
    def __init__(self, x, y, v, m):
        self.x = x
        self.y = y
        self.v = v
        self.m = m

        self.m_shape = np.shape(m)

    @lru_cache()
    def xy_maps(self):
        f = lambda a, b: np.repeat(np.arange(a), b).reshape((a, b))
        return f(self.y, self.x), f(self.x, self.y).T.copy()


    def xy_diffs(self, ym, xm):
        # TODO axis why
        df = lambda m, s, i, a: np.roll(np.abs(np.roll(m - s // 2, - s // 2, axis=a)), i, axis=a)
        return df(ym, self.m_shape[0], self.y, 0), df(xm, self.m_shape[1], self.x, 1)  # t comes for shift type

    def features(self):
        ydf, xdf = self.xy_diffs(*self.xy_maps())

    def relations(self):
        x = [and, or]
        """
        y ~ y | max_y == y0 + 1 <--> y0 = max_y -1
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
