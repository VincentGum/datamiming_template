# Problem 1
# Compare the result
# 5-fold cross validation experiment
# classification accuracy

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score

# load DataSet and generate attribute_set and value_set
data = pd.read_csv('wdbc.data', header=None)
values = data[1]
labels = values.map({'M': 1, 'B': 0})

attributes = data.drop(0, axis=1)
attributes = attributes.drop(1, axis=1)

# apply a simple prepossessing
for i in attributes:
    attributes[i] = attributes[i] - np.mean(attributes[i])

attributes = attributes.values
labels = labels.values

# apply a 5-fold cross validation
accuracy_mean = []
accuracy_ = []

kf = KFold(n_splits=5, shuffle=False, random_state=None)
kf.get_n_splits(attributes)
for train_index, test_index in kf.split(attributes):
    X_train, X_test = attributes[train_index], attributes[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # print(type(X_train))
    # print(X_train.shape)

    accuracy = []

    # using svm
    svm_cls = svm.SVC(kernel='linear')
    svm_cls.fit(X_train, y_train)
    svm_predict = svm_cls.score(X_test, y_test)
    accuracy.append(svm_predict)

    # using logistic regression
    log_cls = LogisticRegression()
    log_cls.fit(X_train, y_train)
    log_accuracy = log_cls.score(X_test, y_test)
    accuracy.append(log_accuracy)

    # using decision tree
    dtree_clf = DecisionTreeClassifier(random_state=0)
    dtree_clf.fit(X_train, y_train)
    dtree_accuracy = dtree_clf.score(X_test, y_test)
    accuracy.append(dtree_accuracy)

    # using gaussian naive bayes
    gnb_clf = GaussianNB()
    gnb_clf.fit(X_train, y_train)
    gnb_accuracy = gnb_clf.score(X_test, y_test)
    accuracy.append(gnb_accuracy)

    # using random forest
    rf_clf = RandomForestClassifier(max_depth=2, random_state=0)
    rf_clf.fit(X_train, y_train)
    rf_accuracy = rf_clf.score(X_test, y_test)
    accuracy.append(rf_accuracy)

    accuracy_.append(accuracy)

for i in range(5):
    accuracy_mean.append((accuracy_[0][i] + accuracy_[1][i] + accuracy_[2][i] + accuracy_[3][i] + accuracy_[4][i])/5)

print(accuracy_mean)

# Problem 2
silhouette = []
calinski_harabaz = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, max_iter=1000, random_state=0).fit(attributes)
    s_score = silhouette_score(attributes, kmeans.labels_, metric='euclidean')
    c_score = calinski_harabaz_score(attributes, kmeans.labels_)
    silhouette.append(s_score)
    calinski_harabaz.append(c_score)

print(silhouette)
print(calinski_harabaz)


