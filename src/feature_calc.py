import numpy as np
from methodtools import lru_cache
from learn import FeatureMatrix
# from

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
        df = lambda m, s, i, a: np.roll(np.minimum(m, m[::-1]+1), i, axis=a)
        return df(ym.copy(), self.m_shape[0], self.y, 0), df(xm.copy().T, self.m_shape[1], self.x, 0).T  # t comes for shift type

    def matrixer(self):
        return np.full(self.m_shape,self.v)

    def features(self):
        ym, xm = self.xy_maps()
        ydf, xdf = self.xy_diffs(ym, xm)
        # self.test()
        # functions = [min, max, avg ==, -, +, *, /, !=, <, >]
        # mt = self.matrixer()
        ma = self.m
        v = self.v

        # print('v', v)

        features = np.array((
            ym < self.y,
            xm < self.x,
            ydf,
            xdf,
            ma == v,
            ma - v,
            # ma < v,
            # ma > v,
            # ma * v,
            # np.abs(ma - v),
            # ma + v
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

        print('YM \n', ym)
        print('XM \n', xm)
        print('ydf \n', ydf)
        print('xdf \n', xdf)

        assert all(ym[0, :]) == 0
        assert all(xm[:, 0]) == 0
        assert all(ydf[self.y, :]) == 0
        assert all(xdf[:, self.x]) == 0

# fm = FeatureMatrix.from_raw_values([[1,2,3],[2,3,4],[5,4,3]])
if __name__ == '__main__':

    fm = FeatureMatrix.from_raw_values(np.ones((4,6)))
    cy, cx = 0, 1
    fc = FeatureCalculator(cy, cx, fm[cy,cx], fm)
    fc.test()
    # print(fc.m)

"""
rules: 
"""

"""
test out:
v'(y',x') = f(y, x, max_y, max_x,...) = ( x > 5) ^ (y < 7) 
... additional features for whole matrix

# TODO: CHECK IF THERE ARE NEW CLASSES APPEARING IN THE DATA

Given task T and train examples (Xin, Xout)ij

Predict output classes of T Y
~classes = Pred_classes(T, Y) = example: (0,1,2,4,8

dla kazdej klasy znajdz funkcje:

v(y,x) == 0 dla f1 @ f2 @f3 ...  or set(background) it means dont evaluate for those pixels. However background has to be optimised for background(a, b) it means size of prediction

v(y, x) = (all, any)(( y > a) ^ (x < b)

facts: y, x, v[y,x] 
vars: Y, X 
compare: <, >, ==, !=
transform: Var {+, -, mod, *, / }
funcs: min, max, wrapped_distance, Counter(obj) --> {'class0':2,'class1':3}
objects: prostokąty ((ay < Y < by), ax < X < bx), V[Y, X] ,~((y0,x0),(y1,x1)...))
        # ~(...) is about pixel (ij) taken out from object and ok for unification
        
        C - comes for complement set 
        O(Y, X, C) :- Count(C) < Y * X / 2

dla v == 0:
    v = background()

dla v == 1:
    v(Y, X) :- 1 <==> S(Y, X)
  

Resources: https://arxiv.org/pdf/1904.11694.pdf 
https://arxiv.org/abs/1808.00508
https://www.wikiwand.com/en/Gram%E2%80%93Schmidt_process
https://www.wikiwand.com/en/Expectation%E2%80%93maximization_algorithm
https://www.wikiwand.com/en/Self-organizing_map
https://github.com/INK-USC/RE-Net
https://link.springer.com/article/10.1007/s10994-017-5668-y
https://arxiv.org/pdf/1611.01989.pdf
https://arxiv.org/pdf/1904.06317.pdf
http://sci.tamucc.edu/~cams/projects/206.pdf
https://arxiv.org/pdf/1711.03243v3.pdf
https://stanford.edu/~jlmcc/papers/SaxeMcCGanguli13CogSciProc.pdf
"""


