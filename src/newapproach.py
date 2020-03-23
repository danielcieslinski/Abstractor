import utils
from learn import Task
from skimage.feature import canny
from numpy import array as a
from feature_calc import FeatureCalculator
import numpy as np

# def main():
tr, te, ev = utils.get_data()
t = Task(298, tr[298])
# t.summary()
# print(t.train[0].input)
# t.plot()
# t.train[0].input[2][4].calc_features(t.train[0].input)

# main()

in_out =[(t.train[i].input.m, t.train[i].output.m) for i in range(len(t.train))]

in1, out1 = in_out[0]

"""
find rules (m1 > 0, m2 > 0, m1-m2 != 0)
"""

"""
cell(0->1) y = (m1 > 0).y 
"""

# https://arxiv.org/pdf/1605.09673v2.pdf
# https://github.com/dbbert/dfn/blob/master/experiment_bouncingMnistOriginal_tensorflow.ipynb

in1_active = np.argwhere(in1 > 0 )
out1_active = np.argwhere(out1 > 0 )
diff = in1 - out1
changed = np.argwhere(diff != 0)

from datetime import datetime
from itertools import product

def aggr(IM):
    features = []

    xsh, ysh = np.shape(IM)

    b1 = datetime.now()
    for y, x in product(range(ysh), range(xsh)):
        c = IM[y][x]
        fc = FeatureCalculator(c.y, c.x, c.v, IM)
        features.append(fc.features().T)
    #     print(y,x)
    #     features[y,x] = fc.features()
    b2 = datetime.now()
    diff = (b1 - b2)
    print('took', diff.microseconds / 1000, 'miliseconds')
    f = np.array(features)
    f = np.reshape(f, (ysh, xsh, *f.shape[1:]))

    return f


print(in1)
# print(changed)
# print(in1_active, out1_active)
y, x = in1_active[0]
in1f = FeatureCalculator(y, x, in1[y,x], in1)

y, x = out1_active[2]
in2f = FeatureCalculator(y, x, in1[y,x], in1)

print(in1f.features() == in2f.features())

agi = aggr(out1)
# ago = aggr(out1)
print(agi.shape)
a = agi[y, x, :].reshape((324))
print(a)
reshaped = agi.reshape((36,-1 ))
print(reshaped.shape)
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(reshaped)

distances, indices = nbrs.kneighbors(reshaped)


print(distances)
print(indices)

# https://github.com/dbbert/dfn/blob/master/experiment_bouncingMnistOriginal_tensorflow.ipynb


"""
If output sizes are the same, size should be treated as given param
When they are different it should be optimized
1. By regression based on the train sample mappings
2. By features in input image
"""