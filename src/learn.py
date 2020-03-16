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

    def __str__(self):
        return str(self.v)

    def __len__(self):
        return len(self.v)


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

    def m_type(self):
        return type(self.m)

    def __str__(self):
        return str(self.m)

    def __call__(self, *args, **kwargs):
        return self.m

    def __len__(self):
        return len(self.m)
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

        self.file_name = None #TODO each task is sorted alphabetically, when loading

        if tofeatured:
            self.train = [self._Example(e) for e in self.raw_train]
            self.test = [self._Example(e) for e in self.raw_test]

    def __raw_data_to_featured(self):
        for e in self.raw_train:
            self.train.append(self._Example(e))

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

        return

    def plot(self):
        utils.plot_task(self.__unparsed())

    class _Example:
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
    t.plot()

if __name__ == '__main__':
    main()


"""
operations: X! * Y!
"""