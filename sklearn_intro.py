from sklearn import datasets, svm

iris = datasets.load_iris()
digits = datasets.load_digits()

# dataset.data = X
# dataset.target = y
# 2D array (n_samples, n_features)

clf = svm.SVC(gamma=0.001, C=100)

clf.fit(digits.data[:-1], digits.target[:-1])
print(clf.predict(digits.data[-1:]))