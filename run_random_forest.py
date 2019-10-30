import numpy as np
from Random_Forest import RandomForestClassifier
from sklearn.model_selection import train_test_split
import Cross_Validation as cv


# data = np.loadtxt('../Assignment1/wine-dataset.csv', delimiter=',', skiprows=1)
data = np.loadtxt('spam.data.txt', delimiter=' ')

x = data[:, :-1]
y = data[:, -1].astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print("Mean for tot dataset: {0}".format(np.mean(y_train)))

rfc = RandomForestClassifier(no_estimators=50, split_measure='entropy',
                             min_impurity_split=0.23, max_depth=None,
                             min_samples_split=2, no_splits=3, max_features='auto',
                             bootstrap=True, random_state=0)
rfc.learn(x_train, y_train)
preds = rfc.classify(x_test)
print("Accuracy: {0}".format(np.mean(y_test == preds)))



# cv_scores = cv.cross_validation(dt, x_train, y_train, k=10)
# print("\nMean Cross-Validation score: {0}".format(np.mean(cv_scores)))


