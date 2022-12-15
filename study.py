import numpy as np
import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import joblib

x_train = np.load("dataset/x_train.npy")
y_train = np.load("dataset/y_train.npy")
x_val = np.load("dataset/x_val.npy")
y_val = np.load("dataset/y_val.npy")

print(x_train.shape)
nsamples, nx, ny = x_train.shape
d2_train_dataset = x_train.reshape((nsamples,nx*ny))

train_data, test_data, train_label, test_label = train_test_split(x_train, y_train)

clf = svm.SVC()
clf.fit(train_data, train_label)

pre = clf.predict(test_data)
ac_score = metrics.accuracy_score(test_label, pre)
print("정답률=", ac_score)


joblib.dump(clf, "eye.pkl")
