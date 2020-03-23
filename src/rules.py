import numpy as np
from methodtools import lru_cache
from learn import FeatureMatrix
from learn import Task
import utils
from toolz import juxt, diff
from operator import sub, truediv

# from pampy import match, _
from fn import _
from collections import Counter
from random import choice
from toolz import compose
from funcy import walk_values, group_by, once
from itertools import  product
from collections import OrderedDict
"""
for each class make V = (i, j) where m[i,j] == v
"""


points = ()
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

    @once
    def produce_all_cords(self):
        ys, xs = self.shape
        ys, xs = range(ys), range(xs)
        return list(product(ys, xs))

    def make_classes(self):
        by_val = lambda yx: self.M[yx]
        self.Tclasses = group_by(by_val, self.all_coords)
        self.Tclasses = walk_values(set, self.Tclasses)

    def how_much_class_changed(self, tc, m2):
        return juxt([truediv, sub])(self.Tclasses[tc], m2.Tclasses[tc])

    def classes_size(self):
        return walk_values(len, self.Tclasses)

    # def class_grouping_refresh(self):
    #     self.make_classes()
    #     self.Tclasses = self.make_classes()
    #     self.tc_sizes = self.classes_size()

    def most_common_class(self):
        return max(self.sizes, key=self.sizes.get)

    # Setters/getters


    #

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

        # else: self.bulk_create(key, value)
        # else: self.
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




class LearningPair:
    def __init__(self, IM, OM):
        self.IM = IM
        self.OM = OM

    def select_init_learning_M(self):
        pass




class UniversalLearner:
    def __init__(self, IMs, OMs):
        """
        :param IMs: [ExtendedMatrix(inp1), ...]
        :param OMs:: [ExtendedMatrix(out1), ...
        """
        self.IMs = IMs
        self.OMs = OMs

    def input_type_checker(self):
        """
        Check if data is format [ [input_matrices, output_matricies] ]
        or [ [IM1, OM1] ... ]
        :return: wrapper around *args **kwargs
        """

class BackgroundIndicator(UniversalLearner):
    def __init__(self, IMs, OMs):
        super(BackgroundIndicator, self).__init__(IMs, OMs)

    def naive_solve(self):
        pass

    def color(self):
        pass




class EasyAgent:
    # Without objects
    def __init__(self, IM, OM, learned=None, to_learn=None, classes=None):
        self.IM = IM
        self.OM = OM
        self.L = learned
        self.to_learn = to_learn
        self.classes = classes


    def find_background(self):
        return max(self.classes)

    def process_input_matrix(self):
        # Find
        for c in self.classes:
            pass

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

    em = ExtendedMatrix(in1)
    em.test()


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

