print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

a= np.zeros((3,4))

b= np.ones((3,4))

print(a)
print(b)


c = np.concatenate((a, b), 1)
print(c)

print(len(c))
print(len(c[0]))


