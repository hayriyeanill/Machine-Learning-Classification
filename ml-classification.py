import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

df = pd.read_csv("bank-additional-full.csv",sep=";")

le = LabelEncoder()
job_labels = le.fit_transform(df['job'])
marital_labels = le.fit_transform(df['marital'])
education_labels = le.fit_transform(df['education'])
default_labels = le.fit_transform(df['default'])
housing_labels = le.fit_transform(df['housing'])
loan_labels = le.fit_transform(df['loan'])
contact_labels = le.fit_transform(df['contact'])
month_labels = le.fit_transform(df['month'])
day_of_week_labels = le.fit_transform(df['day_of_week'])
poutcome_labels = le.fit_transform(df['poutcome'])
y_labels = le.fit_transform(df['y'])

df['job'] = job_labels
df['marital'] = marital_labels
df['education'] = education_labels
df['default'] = default_labels
df['housing'] = housing_labels
df['loan'] = loan_labels
df['contact'] = contact_labels
df['month'] = month_labels
df['day_of_week'] = day_of_week_labels
df['poutcome'] = poutcome_labels
df['y'] = y_labels

x = df.iloc[:,:-1]  # independent columns
y = df.iloc[:,20]   # target column

# backward elimination
import statsmodels.api as sm
x = np.append(arr = np.ones((41188, 1)).astype(float), values = x, axis = 1) # adding column for x0
x_opt = x[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12,13,14,15,16,17,18,19,20]]
regressor_ols = sm.OLS(y, x_opt).fit()

sl = 0.05
numVars = len(x_opt[0])
for i in range(0, numVars):
    regressor_ols = sm.OLS(y, x_opt).fit()
    print(regressor_ols.summary())
    maxVar = max(regressor_ols.pvalues)
    if maxVar > sl:
        for j in range(0, numVars - i):
            if (regressor_ols.pvalues[j] == maxVar):
                x_opt = np.delete(x_opt, j, 1)

# K-Fold Cross Validation
kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)

for train_index, test_index in kf.split(x_opt):
      X_train, X_test = x_opt[train_index], x_opt[test_index]
      y_train, y_test = y[train_index], y[test_index]

# feature scaling
from sklearn.preprocessing import StandardScaler
scx = StandardScaler()
x_train = scx.fit_transform(X_train)
x_test = scx.fit_transform(X_test)

def visualize_confussion_matrix(conf_matrix, cmap):
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in conf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in
                         conf_matrix.flatten() / np.sum(conf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(conf_matrix, annot=labels, fmt="", cmap=cmap)
    plt.show()

# logistic regression with feature scaling
from sklearn.linear_model import LogisticRegression
lrc = LogisticRegression(random_state = 0)
lrc.fit(x_train, y_train)
lrc_pred = lrc.predict(x_test)
cm_lrc = confusion_matrix(y_test, lrc_pred)
acc_lrc = accuracy_score(y_test, lrc_pred)
plt.title("Logistic Regression Confussion Matrix")
visualize_confussion_matrix(cm_lrc, 'Blues')
print("Logistic Regression", acc_lrc)

# decision tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth = 3, criterion = 'entropy', random_state = 0)
dtc.fit(X_train, y_train)
dtc_pred = dtc.predict(X_test)
cm_dtc = confusion_matrix(y_test, dtc_pred)
acc_dtc = accuracy_score(y_test, dtc_pred)
plt.title("Decision Tree Confussion Matrix")
visualize_confussion_matrix(cm_dtc, 'Greens')
print("Decision Tree", acc_dtc)

# random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 50, max_depth = 5, criterion = 'entropy', random_state = 0)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
cm_rfc = confusion_matrix(y_test, rfc_pred)
acc_rfc = accuracy_score(y_test, rfc_pred)
plt.title("Random Forest Confussion Matrix")
visualize_confussion_matrix(cm_rfc, 'Reds')
print("Random Forest", acc_rfc)

# kNN with feature scaling
from sklearn.neighbors import KNeighborsClassifier
error_rate = []
for i in range(1, 11):
   knn = KNeighborsClassifier(n_neighbors = i, metric = 'minkowski', p = 2)
   knn.fit(x_train, y_train)
   knn_pred = knn.predict(x_test)
   error_rate.append(np.mean(knn_pred != y_test))

plt.plot(range(1,11),error_rate,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. k Value')
plt.xlabel('k')
plt.ylabel('Error Rate')
plt.show()

# Optimum k value
knn = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 2)
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)
cm_knn = confusion_matrix(y_test, knn_pred)
plt.title("K-NN Confussion Matrix")
acc_knn = accuracy_score(y_test, knn_pred)
visualize_confussion_matrix(cm_knn,'Purples')
print("K-NN", acc_knn)


# SVC with feature scaling non-linear
from sklearn.svm import SVC
svm = SVC(kernel = 'rbf', random_state = 0)
svm.fit(x_train, y_train)
svm_pred = svm.predict(x_test)
acc_svm = accuracy_score(y_test, svm_pred)
cm_svm = confusion_matrix(y_test,svm_pred)
plt.title("SVM Confussion Matrix")
visualize_confussion_matrix(cm_svm, 'Oranges')
print("SVM", acc_svm)

# k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies_lrc = cross_val_score(estimator = lrc, X = x_train, y = y_train, cv = 10)
accuracies_dtc = cross_val_score(estimator = dtc, X = X_train, y = y_train, cv = 10)
accuracies_rfc = cross_val_score(estimator = rfc, X = X_train, y = y_train, cv = 10)
accuracies_knn = cross_val_score(estimator = knn, X = x_train, y = y_train, cv = 10)
accuracies_svm = cross_val_score(estimator = svm, X = x_train, y = y_train, cv = 10)
print("Logistic Regression Accuracy Mean: ", accuracies_lrc.mean(), "and Variance: ", accuracies_lrc.std())
print("Decision Tree Accuracy Mean: ", accuracies_dtc.mean(), "and Variance: ", accuracies_dtc.std())
print("Random Forest Accuracy Mean: ", accuracies_rfc.mean(), "and Variance: ", accuracies_rfc.std())
print("KNN Accuracy Mean: ", accuracies_knn.mean(), "and Variance: ", accuracies_knn.std())
print("SVM Accuracy Mean: ", accuracies_svm.mean(), "and Variance: ", accuracies_svm.std())


