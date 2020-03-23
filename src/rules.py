import numpy as np
from methodtools import lru_cache
from learn import FeatureMatrix
from learn import Task
import utils
from toolz import juxt, diff
from operator import sub, truediv

tr, te, ev = utils.get_data()
t = Task(298, tr[298])

in_out =[(t.train[i].input.m, t.train[i].output.m) for i in range(len(t.train))]
in1, out1 = in_out[0]
#
# def background(in1, out1):
#     unique, counts = np.unique(out1, return_counts=True)
#     c = dict(zip(unique, counts))
#     if in1.shape == out1.shape: return in1.shape

# from pampy import match, _
from fn import _
from collections import Counter
from random import choice
from toolz import compose
from funcy import walk_values, group_by
"""
for each class make V = (i, j) where m[i,j] == v
"""

# class Point:
#     def __init__(self):
from itertools import  product

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
        self.class_grouping_refresh()
        self.sizes = self.classes_size()

    def produce_all_cords(self):
        ys, xs = self.shape
        ys, xs = range(ys), range(xs)
        return list(product(ys, xs))

    def make_classes(self):
        by_val = lambda yx: self.M[yx]
        self.Tclasses = group_by(by_val, self.all_coords)

    def how_much_class_changed(self, tc, m2):
        return juxt([truediv, sub])(self.Tclasses[tc], m2.Tclasses[tc])

    def classes_size(self):
        return walk_values(len, self.Tclasses)

    def class_grouping_refresh(self):
        self.make_classes()
        # self.Tclasses = self.make_classes()
        self.tc_sizes = self.classes_size()

    def most_common_class(self, recallc=True):
        if recallc:
            self.class_grouping_refresh()
            return max(self.tc_sizes, key=self.tc_sizes.get)

    def __getitem__(self, item):
        return self.M[item]

    def __setitem__(self, key, value):
        self.M[key] = value

    def test(self):
        print(self.M)
        print(self.all_coords)
        print(self.Tclasses)
        print(self.most_common_class())

em = ExtendedMatrix(in1)
em.test()


class TrainPair:
    def __init__(self, IM, OM):
        self.IM = IM
        self.OM = OM

class RuleMap:
    def __init__(self, pairs):
        self.pairs = pairs


# class ValueClass:
#     def __init__(self, v, is_background, nodes):
#         self.v = v
#         self.is_background = is_background
#         self.nodes = nodes
#         # self.size()
#
#     # def size(self, x=self):
#     #     return len(x.vcc_agg[self.v])
#
#     def how_much_changed(self, M1, M2):
#         """
#         :param M1:
#         :param M2:
#         :return: len(v) / len(v') , len(v) - (v')
#         """
#         return compose(juxt([truediv, sub]), f)(M1, M2)

class EasyAgent:
    # Without objects
    def __init__(self, IM, OM, learned=None, to_learn=None, classes=None):
        self.IM = IM
        self.OM = OM
        self.L = learned
        self.to_learn = to_learn
        self.classes = classes

        if self.L == None:
            self.L = np.empty([])

        if self.classes == None:
            self.classes = np.array([ValueClass(v, None, self.IM) for v in range(10)])

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


a = EasyAgent(in1, out1, None, None)


"""
Dla każdej klasy 0~9

1. Select object O( max-min(Y|= (v[y,_] == i), max-min(x|=v[_,x] == i) ) = O(Y°,X°) gdzie var° is fixed a max-min oznacza tuple (min(var), max(var)) 
2. Transform O by functions: 

"""

