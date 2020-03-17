import utils
from learn import Task
from skimage.feature import canny
from numpy import array as a
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

print(changed)
print(in1_active, out1_active)

# https://github.com/dbbert/dfn/blob/master/experiment_bouncingMnistOriginal_tensorflow.ipynb
