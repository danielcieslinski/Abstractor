import numpy as np
import utils
from methodtools import lru_cache

"""
def find_func(inp, out):
    find_features
    match_features
"""

class Cell:
    def __init__(self, x, y, v):
        self.x = x
        self.y = y
        self.v = v

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
        if not isinstance(size, (tuple, list)):
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

    def __call__(self, *args, **kwargs):
        pass


    #Just return value of main numpy array
    def __setitem__(self, item, value):
            self.m[item] = value

    def __getitem__(self, item):
            return self.m[item]

class Task:
    def __init__(self,index, task, tofeatured=True):
        self.index = index
        self.raw_train = task['train']
        self.raw_test = task['test']

        # Corrseponding pairs
        self.train_inp = []
        self.train_out = []


        self.file_name = None #TODO each task is sorted alphabetically, when loading
        if tofeatured: self._raw_matrix_to_featured() #In place

    def _raw_matrix_to_featured(self):
        for i in range(len(self.train)):
            print(self.train[i])
            self.train
            self.train = list(map(FeatureMatrix.from_raw_values, self.train))


def main():
    tr, te, ev = utils.get_data()
    Task(298, tr[298])

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




if __name__ == '__main__':
    # test_matrix()
    main()


"""
operations: X! * Y!

"""