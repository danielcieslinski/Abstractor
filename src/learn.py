import numpy as np
import utils
from methodtools import lru_cache
from pprint import pprint

"""
def find_func(inp, out):
    find_features
    match_features
"""

class Cell:
    def __init__(self, y, x, v):
        self.y = y
        self.x = x
        self.v = v
        """
        Feature vector
        0: x
        1: y
        """
        self.fv = {}

    def calc_features(self, matrix):
        """
        :param matrix: Numpy of array matrices(FeaturedMatrix.m)
        :return:
        """
        y, x = np.shape(matrix)

        @lru_cache()
        def xy_maps():
            f = lambda a, b: np.repeat(np.arange(a), b).reshape((a, b))
            return f(y, x), f(x, y).T.copy()


        def xy_diffs(ym, xm):
            # TODO axis why
            df = lambda m, s, i, a: np.roll(np.abs(np.roll(m - s // 2, - s//2, axis=a)), i, axis=a)
            return df(ym, y, self.y, 0), df(xm, x, self.x, 1) #t comes for shift type

        # def test():
        #     # xy_maps_tests
        #     assert all(ym[0, :]) == 0
        #     assert all(xm[:, 0]) == 0
        #     assert all(ydf[self.y, :]) == 0
        #     assert all(xdf[self.x, :]) == 0

        ydf, xdf = xy_diffs(*xy_maps())



    def __call__(self, *args, **kwargs):
        return self.v

    def __str__(self):
        return str(self.v)

    def __len__(self):
        return len(self.v)

    def __add__(self, other):
        return self.v + other

    def __matmul__(self, other):
        return self.v * other

    def __copy__(self):
        return Cell(self.x, self.y, self.v)

    def __deepcopy__(self, memodict={}):
        return Cell(self.x, self.y, self.v.copy())

    def copy(self):
        return self.__copy__()

class FeatureMatrix:
    """
    Call example:
    fm = FeatureMatrix.from_raw_values([[1,2,3],[2,3,4],[5,4,3]])
    """

    def __init__(self, m=None):
        self.m = m

    @classmethod
    def from_raw_values(cls, arr, method='iter'):
        if method == 'iter': return cls(m=cls.ndarr_to_FM_iterative(arr))

    @classmethod
    def empty(cls, size, default_val=None):
        if not isinstance(size, (tuple, list)):# TODO Better check
            raise ValueError('Supporting only 2D matrices')
        y, x = size
        if isinstance(x, int) and isinstance(y, int):
            return cls(m=np.empty(size, default_val))

        raise ValueError('Values of tuples have to be Integers')

    @staticmethod
    def ndarr_to_FM_iterative(arr):
        """
        Converts numpy array of int values to FeatureMatrix

        :param arr: numpy array 2D only at the moment
        :return: Feature matrix of cells
        """
        ym, xm = np.shape(arr)
        out = np.empty((ym, xm), dtype=Cell)
        for y in range(ym):
            for x in range(xm):
                out[y][x] = Cell(y, x, arr[y][x])

        return out

    def m_type(self):
        return type(self.m)


    # def shape(self):
        # return len(self.m)

    def __str__(self):
        return str(self.m)

    def __call__(self, *args, **kwargs):
        return self.m

    def __len__(self):
        return len(self.m)

    def __deepcopy__(self, memodict={}):
        return FeatureMatrix(m=self.m.deepcopy())

    #Just return value of main numpy array
    def __setitem__(self, item, value):
            self.m[item] = value

    def __getitem__(self, item):
            return self.m[item]
    # math
    def __add__(self, other):
        return self.m + other

class Task:
    def __init__(self,index, task, tofeatured=True):
        self.index = index
        self.raw_train = task['train']
        self.raw_test = task['test']

        self.file_name = None #TODO each task is sorted alphabetically, when loading

        if tofeatured:
            self.train = [self.__Example(e) for e in self.raw_train]
            self.test = [self.__Example(e) for e in self.raw_test]

    def __raw_data_to_featured(self):
        for e in self.raw_train:
            self.train.append(self.__Example(e))

    def __unparsed(self):
        return {'train': self.raw_train, 'test': self.raw_test}

    def summary(self):
        print('Task number:', self.index)
        print('Number of train examples',len(self.train))

        print(self.train[0].input[0,1].v)

        input_shapes, output_shapes = [], []

        for e in self.train:
            input_shapes.append(np.shape(e.input))
            output_shapes.append(np.shape(e.output))

        print('    ', 'input shapes', input_shapes)
        print('    ', 'output shapes', output_shapes)

    def plot(self):
        utils.plot_task(self.__unparsed())

    class __Example:
        def __init__(self, e, to_featured=True):
            self.input = e['input']
            self.output = e['output']

            if to_featured:
                self.data_to_featured()

        def data_to_featured(self):
            self.input = FeatureMatrix.from_raw_values(self.input)
            self.output = FeatureMatrix.from_raw_values(self.output)




def test_matrix():
    F = FeatureMatrix.empty((2,3))
    # F = FeatureMatrix()
    fm = FeatureMatrix.from_raw_values([[1,2,3],[2,3,4],[5,4,3]])
    # F = FeatureMatrix(m=[1,2])
    F[0,1] = 1
    print(F.m)

def find_features(c1, c2):
    # if len(c1) != 1 or len(c2) != 1
    #     map(find_features, )

    """
    min_val, max_val, min_x, max_x, min_y, max_y

    funcs beetwen two
    min, max, avg ==, -, +, *, /, !=, <, >
    """

    print(c1, c2)

def main():
    tr, te, ev = utils.get_data()
    t = Task(298, tr[298])
    t.summary()
    # print(t.train[0].input)
    # t.plot()
    t.train[0].input[2][4].calc_features(t.train[0].input)

if __name__ == '__main__':
    main()


"""
operations: X! * Y!
"""