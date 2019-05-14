'''This project was done for Machine learning class at Morgan State University. The Breast Cancer Wisconsin (Diagnostic)
Data Set was obtained from UCI Machine Learning Repository. This model has an accuracy of 97.7% using the sklearn SVM
with rbf kernel.'''
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import statistics
import pandas as pd
from warnings import filterwarnings

filterwarnings('ignore')

false_rate = {('M', 'B'): 0, ('B', 'M'): 0}
file_name = 'data.xlsx'

df = pd.read_excel(file_name, usecols=range(1, 33))
x = df.loc[:, df.columns != 'diagnosis']
y = df['diagnosis']


def model(x_train, x_test, y_train, y_test):
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)

    # poly kernel yields 90% accuracy
    # rbf kernel yields 97.7% accuracy
    # linear kernel yields 96.4 accuracy
    svm = SVC(kernel='rbf')
    svm.fit(x_train_std, y_train)

    y_pred = svm.predict(x_test_std)  # predicted answer using model
    y_test_list = y_test.tolist()

    # d contains all of the prediction and weather they were correctly classified or not
    prediction_bool_lst = [y_test_list[i] == y_pred[i] for i in range(0, len(y_pred))]
    d = {'Expected': y_test_list, 'Predicted': y_pred.tolist(), 'Matched': prediction_bool_lst}
    test_result = pd.DataFrame(data=d)

    # calculating the rate of false positives and false negatives
    for i in range(0, test_result.size // 3):
        row = test_result.iloc[i]
        if not row['Matched']:
            false_prediction = (row['Expected'], row['Predicted'])
            false_rate[false_prediction] += 1

    return accuracy_score(y_pred, y_test)


k = 10
kf = KFold(n_splits=k)
kf.get_n_splits(x)
total_accuracy = []
for train, test in kf.split(x):
    x_split_train, x_split_test, y_split_train, y_split_test = x.iloc[train], x.iloc[test], y.iloc[train], y.iloc[test]
    total_accuracy.append(model(x_split_train, x_split_test, y_split_train, y_split_test))

for j in range(0, len(total_accuracy)):
    print(f'set {j + 1}:', total_accuracy[j])
print('Total Accuracy:', statistics.mean(total_accuracy))
