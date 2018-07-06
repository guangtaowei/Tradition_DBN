from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import xlrd
import numpy as np

# X, Y = make_regression(n_samples=10, n_features=4, n_informative=1, random_state=0, shuffle=False)

path_data = "data/airdata.xlsx"

data = xlrd.open_workbook(path_data)
table = data.sheet_by_index(0)
data_num = table.nrows - 1

temperature = table.col_values(8)[1:]
temperature = np.array(temperature)
wind = table.col_values(9)[1:]
wind = np.array(wind)
weather = table.col_values(10)[1:]
weather = np.array(weather)
moisture = table.col_values(11)[1:]
moisture = np.array(moisture)

X = []
X.append(temperature)
X.append(wind)
X.append(weather)
X.append(moisture)
X = np.array(X)
X = np.transpose(X)
print(X.shape)

pm25 = table.col_values(2)[1:]
Y = np.array(pm25)

print(Y.shape)
# print(Y)

'''X_train = [[1, 2, 3, 4, 5], [1, 4, 2, 7, 1], [5, 7, 9, 2, 5]]
Y_train = [3, 2, 9]

X_test = [[2, 3, 4, 5, 6], [6, 3, 1, 6, 7], [3, 5, 8, 1, 6]]
Y_test = [4, 1, 8]'''

regr = RandomForestRegressor().fit(X, Y)
print("RandomForestRegressor.feature_importances_:\n", regr.feature_importances_)

cca = CCA().fit(X, Y)
print("cca.x_weights_:\n", cca.x_weights_)
# print("cca.x_loadings_:\n", cca.x_loadings_)
# print("cca.x_scores_:\n", cca.x_scores_)

'''X_PCA = PCA().fit_transform(X_train)
X_CCA, Y_CCA = CCA().fit(X_train, Y_train).transform(X_train, Y_train)
X_CCA_test, Y_CCA_test = CCA().fit(X_test, Y_test).transform(X_test, Y_test)
print("X_PCA:\n", X_PCA)
print("X_CCA:\n", X_CCA, "Y_CCA:\n", Y_CCA)

reg_origin = AdaBoostRegressor().fit(X_train, Y_train)
reg_PCA = AdaBoostRegressor().fit(X_PCA, Y_train)
reg_CCA = AdaBoostRegressor().fit(X_CCA, Y_train)

print("\nreg_origin:\t", reg_origin.score(X_test, Y_test), "\n", reg_origin.predict(X_test))
print("\nreg_PCA:\t", reg_PCA.score(PCA().fit_transform(X_test), Y_test), "\n",
      reg_PCA.predict(PCA().fit_transform(X_test)))
print("\nreg_CCA:\t", reg_CCA.score(X_CCA_test, Y_test), "\n", reg_CCA.predict(X_CCA_test))'''
