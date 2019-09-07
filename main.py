import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()
#classifire
clf = svm.SVC(gamma=0.1, C=100)
print(len(digits.data))

x,y = digits.data[:-1], digits.target[:-1]
clf.fit(x,y)

print('Prediction:', clf.predict(digits.data[:-1]))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()