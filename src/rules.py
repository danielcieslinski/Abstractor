import numpy as np
from methodtools import lru_cache
from learn import FeatureMatrix, Task
import utils
from toolz import juxt, diff, reduce
from operator import sub, truediv

from random import choice
from toolz import compose
from funcy import walk_values, group_by, once
from itertools import  product
"""
for each class make V = (i, j) where m[i,j] == v
"""
lm = lambda F, args: list(map(F, *args))

# c = lambda M: [list(map(lambda y,x, M : (y,x) if  M[y,x] == v else None, M) for v in range(10) )]

# def empty_dict(self):
#     return {k: [] for k in range(10)}

class ExtendedMatrix:
    def __init__(self, m):
        self.M = m
        self.shape = self.M.shape

        self.Tclasses = None #dict {0:[(y1,x1)...]}
        self.all_coords = self.produce_all_cords()

        # --------------
        self.chain = []

        # --------------
        #Only init call
        # self.class_grouping_refresh()
        self.make_classes()
        self.sizes = self.classes_size()
        # --------------

    # @once
    def produce_all_cords(self):
        ys, xs = self.shape
        ys, xs = range(ys), range(xs)
        return list(product(ys, xs))

    def make_classes(self):
        by_val = lambda yx: self.M[yx]
        self.Tclasses = group_by(by_val, self.all_coords)
        self.Tclasses = walk_values(set, self.Tclasses)

    def mdiff(self, tc, m2):
        return juxt([truediv, sub])(self.Tclasses[tc], m2.Tclasses[tc])

    def classes_size(self):
        return walk_values(len, self.Tclasses)

    def most_common_class(self):
        return max(self.sizes, key=self.sizes.get)

    # Setters/getters
    def __getitem__(self, item):
        return self.M[item]

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == 2:
            yk, xk = key
            if isinstance(yk, int) and isinstance(xk, int):
                #TODO update class aggregation
                tmp = self.M[key]
                self.Tclasses[tmp].discard(key)
                self.Tclasses[value].add(key)
                self.M[key] = value


    def __copy__(self):
        n = ExtendedMatrix(m=self.M.copy())
        n.chain = self.chain.copy()
        return n

    # Test
    def test(self):
        print('Printing matrix')
        print(self.M)
        print('Printing all coords')
        print(self.all_coords)
        print('Printing classes aggregation')
        print(self.Tclasses)
        print('The most common class is:')
        print(self.most_common_class())
        print('Printing classess sizes')
        print(self.sizes)
        print('Changing (0,0)')
        self[0,0] = 3
        print(self.M)
        print(self.Tclasses)

from skimage.feature import hog

class EasyAgent:
    # Without objects
    def __init__(self, IM, OM, pair_num):
        self.IM = IM
        self.OM = OM
        # self.pair_num = pair_num
        #
        # #
        # self.bg_color = self.find_background()
        # self.L = np.full(self.OM.shape(), self.bg_color)
        # self.objects = []

        print(self.IM.M)

        fd, hog_image = hog(self.OM.M, orientations=8, pixels_per_cell=(2, 2),
                            cells_per_block=(1, 1), visualize=True, multichannel=False)
        print(fd)

    def find_background(self):
        return self.IM.most_common_class()

    def make_object(self, c):
        ay = min(self.IM.Tclasses[c][0], key=lambda t: t[0] )
        by = max(self.IM.Tclasses[c][0], key=lambda t: t[0] )

        ax = min(self.IM.Tclasses[c][1], key=lambda t: t[1] )
        bx = max(self.IM.Tclasses[c][1], key=lambda t: t[1] )

        yr, xr = range(ay, by+1), range(ax, bx+1)
        combs = product(yr, xr)
        reduce(lambda yx: self.L[yx].set(c), combs)
        self.objects.append()

    def process_input_matrix(self):
        for c1 in self.IM.Tclasses:
            if self.IM.Tclasses[c1] == self.bg_color: pass
            self.make_object(c1)

    def get_available_moves(self):
        """
        set color as background
        :return:
        """
        """
        (vars, (compare, transform),
        :return:
        """
        pass

    def chooser(self, Vars, F):
        return list(map(choice, [Vars, F]))


def main():
    tr, te, ev = utils.get_data()
    t = Task(298, tr[298])

    in_out = [(t.train[i].input.m, t.train[i].output.m) for i in range(len(t.train))]
    in1, out1 = in_out[0]

    emi = ExtendedMatrix(in1)
    emo = ExtendedMatrix(out1)

    # print(emo.test())

    EasyAgent(emi, emo, None)

if __name__ == '__main__':
    main()

"""
Dla każdej klasy 0~9

1. Select object O( max-min(Y|= (v[y,_] == i), max-min(x|=v[_,x] == i) ) = O(Y°,X°) gdzie var° is fixed a max-min oznacza tuple (min(var), max(var)) 
2. Transform O by functions: 

"""

"""
https://arxiv.org/pdf/1803.05252.pdf
Algebr auczenie

https://arxiv.org/pdf/1802.09436.pdf
lololo

https://link.springer.com/content/pdf/10.1007/s13163-018-0273-6.pdf
https://www.wikiwand.com/en/Algebraic_geometry
https://www.wikiwand.com/en/Gr%C3%B6bner_basis
"""

