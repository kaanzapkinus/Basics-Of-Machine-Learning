import sklearn
import matplotlib.pyplot as plt
from sklearn import tree
print(sklearn.__version__)
X = [[20, 100], [30, 80], [35, 60], [20, 60], [10, 50], [0, 10], [0, 100]]
Y = [1,0,0,1,1,0,0]
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, Y)
tree.plot_tree(clf)
plt.show()

print(clf.predict([[29, 100]]))
