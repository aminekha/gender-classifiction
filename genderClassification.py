"""
This program uses scikit-learn modules to predict whether a person is male or female from data (height, weight, shoes size)
"""
from sklearn import tree # we will use the tree module

# [height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

# Label for the data (whether X is male or female)
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Define a variable to store our decision tree module
clf = tree.DecisionTreeClassifier()

# The fit method train the decision tree on our data set
clf = clf.fit(X, Y)

# The module is now trained let's try to predict
prediction = clf.predict([[190, 70, 43], [160, 60, 38]])
print(prediction)

