import numpy as np
import utils
from methodtools import lru_cache

"""
def find_func(inp, out):
    find_features
    match_features
"""

D = {}


def test_matrix():
    F = FeatureMatrix.from_raw_values(np.array([[1,2,3],[2,3,4],[5,4,3]]))
    # print(F.m)
    # print(F)
    F[0,1] = 1
    # print(m[0,1])
    # print(m.m)
    # print(F[0,1])
    print(F.m)


class Cell:
    def __init__(self, x, y, v):
        self.x = x
        self.y = y
        self.v = v

class FeatureMatrix:
    def __init__(self, m=None):
        self.m = m

    # @lru_cache()
    @classmethod
    def from_raw_values(cls, arr, method='iter'):
        if method == 'iter': return cls(m=cls.ndarr_to_FM_iter(arr))

    @classmethod
    def empty(cls, size, default_val=None):
        if not isinstance(size, tuple):
            raise ValueError('Supporting only 2D matrices')
        y, x = size
        if isinstance(x, int) and isinstance(y, int):
            return cls(m=np.empty(size, default_val))

        raise ValueError('Values of tuples have to be Integers')

    # @lru_cache()
    @staticmethod
    def ndarr_to_FM_iter(arr):
        """
        :param arr: numpy array 2D only at the moment
        :return: Feature matrix of cells
        """
        ym, xm = np.shape(arr)
        out = np.empty((ym, xm), dtype=Cell)
        for y in range(ym):
            for x in range(xm):
                out[y][x] = Cell(y, x, arr[y][x])

        return out


    #Just return value of main numpy array
    def __setitem__(self, item, value):
            self.m[item] = value

    def __getitem__(self, item):
            return self.m[item]

    # def __setitem__(self, item, value):
    #     if isinstance(item, tuple):
    #         y, x = item
    #         self.m[y][x] = value
    #
    # def __getitem__(self, item):
    #     if isinstance(item, tuple):
    #         y, x = item
    #         return self.m[y][x]


class Task:
    def __init__(self,index, task, tofeatured=True):
        self.index = index
        self.train = task['train']
        self.test = task['test']

        self.file_name = None #TODO each task is sorted alphabetically, when loading

        if tofeatured: self._raw_matrix_to_featured() #In place

    def _raw_matrix_to_featured(self):
        self.train = list(map(FeatureMatrix, self.train))



def find_features(c1, c2):
    # if len(c1) != 1 or len(c2) != 1
    #     map(find_features, )

    """
    min_val, max_val, min_x, max_x, min_y, max_y

    funcs beetwen two
    min, max, avg ==, -, +, *, /, !=, <, >
    """

    print(c1, c2)


def calc_features(m):
    """
    :param m: Matrix
    :return:
    """
    m = np.array(m)
    print(np.shape(m))

    # x = map(find_features, m) #for_row
    # x = map(find_features, x) #for cell

    ym, xm = np.shape(m)
    print(ym)

    for y1 in range(ym):
        for x1 in range(xm):
            for y2 in range(ym):
                for x2 in range(xm):
                    find_features([m[y1][x1], y1, x1], [m[y2][x2], y2, x2])




def solve_task(task):
    # utils.plot_task(task)
    tr_samples = task['train']
    # print(s0)
    print(tr_samples[0])
    calc_features(tr_samples[0]['input'])


def main():
    tr, te, ev = utils.get_data()
    solve_task(tr[298])


if __name__ == '__main__':
    test_matrix()
    # a = np.array([1,2,3,4])
    # print(a-5)


    # main()

"""
operations: X! * Y!

"""