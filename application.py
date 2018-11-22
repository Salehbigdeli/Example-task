'''Run application and give input features in standard input one by one.'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import pickle
import numpy as np

cols = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13']
clf = joblib.load('trained_random_forrest_v1.joblib')
with open('means_for_v1', 'rb') as f:
    stats = pickle.load(f)

inp = []
for c in cols:
    val = input('Enter ' + c)
    if val == 'n':
        val = stats[c]
    else:
        val = float(val)
    inp.append(val)

print(clf.predict(np.array(inp).reshape(1, -1)))
