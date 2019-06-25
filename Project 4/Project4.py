"""
    Kushagra Rastogi
    304640248
    ECE 219
    Project 4
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


############################################################################### DATASET 1 ############################################################################################

##### QUESTION 1

net_back = pd.read_csv("network_backup_dataset.csv")

backup = net_back.groupby([net_back["Week #"],net_back["Day of Week"],net_back["Work-Flow-ID"]],sort=False).sum().iloc[:,1:2]
wf0 = []; wf1 = []; wf2 = []; wf3 = []; wf4 = []
for n,i in backup.iterrows():
    if n[2] == 'work_flow_0':
        wf0.append(i[0])
    elif n[2] == 'work_flow_1':
        wf1.append(i[0])
    elif n[2] == 'work_flow_2':
        wf2.append(i[0])
    elif n[2] == 'work_flow_3':
        wf3.append(i[0])
    elif n[2] == 'work_flow_4':
        wf4.append(i[0])

x = np.arange(0,105,dtype=int)+1
fig, ax = plt.subplots()
ax.plot(x,wf0, label='work_flow_0')
ax.plot(x,wf1, label='work_flow_1')
ax.plot(x,wf2, label='work_flow_2')
ax.plot(x,wf3, label='work_flow_3')
ax.plot(x,wf4, label='work_flow_4')
ax.legend(loc='best')
plt.xlabel("Number of days"); plt.ylabel("Backup Size (GB)"); plt.title("Backup Sizes for all workflows")
plt.show()


##### QUESTION 2A
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression

net_back = net_back.drop(columns=['Backup Time (hour)'])

X = OrdinalEncoder().fit_transform(net_back.iloc[:,0:5].values)
y = net_back.iloc[:,5].values

cv_results = cross_validate(LinearRegression(),X,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
train_rmse = np.sqrt(cv_results['train_score']*(-1.))
test_rmse = np.sqrt(cv_results['test_score']*(-1.))
print("Train RMSE for Linear Regression for each fold in 10-fold cross validation: \n", train_rmse)
print("Test RMSE for Linear Regression for each fold in 10-fold cross validation: \n", test_rmse)

best_est = cv_results['estimator'][9]
y_pred = best_est.predict(X)
x = np.arange(len(y))+1

fig, ax = plt.subplots()
ax.plot(x,y, 'ro', label='True')
ax.plot(x,y_pred, 'bo', label='Fitted')
ax.legend(loc='best')
plt.xlabel("Data Point"); plt.ylabel("Backup Size (GB)"); plt.title("Linear Regression Fitted VS True")
plt.show()

res = y - y_pred
fig, ax = plt.subplots()
ax.plot(x,res, 'ro', label='Residual')
ax.plot(x,y_pred, 'bo', label='Fitted')
ax.legend(loc='best')
plt.xlabel("Data Point"); plt.ylabel("Backup Size (GB)"); plt.title("Linear Regression Fitted VS Residuals")
plt.show()


##### QUESTION 2B
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn.model_selection import KFold
import graphviz

X = OrdinalEncoder().fit_transform(net_back.iloc[:,0:5].values)
y = net_back.iloc[:,5].values

import warnings
warnings.filterwarnings("ignore")

rf = RandomForestRegressor(n_estimators=20,max_depth=4,bootstrap=True,max_features=5,oob_score=True)
cv_results = cross_validate(rf,X,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
train_rmse = np.mean(np.sqrt(cv_results['train_score']*(-1.)))
test_rmse = np.mean(np.sqrt(cv_results['test_score']*(-1.)))
print("Average Train RMSE for Random Forest Regression: ", train_rmse)
print("Test RMSE for Random Forest Regression: ", test_rmse)

oob_score = [i.oob_score_ for i in cv_results['estimator']]
print("Average Out of Bag error: ", 1-np.mean(oob_score))

trees = np.arange(1,201)
oob1 = []; oob2 = []; oob3 = []; oob4 = []; oob5 = [];
test_rmse1 = []; test_rmse2 = []; test_rmse3 = []; test_rmse4 = []; test_rmse5 = [];
for i in trees:
    print(i)
    rf = RandomForestRegressor(n_estimators=i,max_depth=4,bootstrap=True,max_features=1,oob_score=True,warm_start=True)
    cv_results = cross_validate(rf,X,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
    test_rmse1.append( np.mean(np.sqrt(cv_results['test_score']*(-1.))) )
    oob1.append( 1-np.mean([i.oob_score_ for i in cv_results['estimator']]) )
    rf = RandomForestRegressor(n_estimators=i,max_depth=4,bootstrap=True,max_features=2,oob_score=True,warm_start=True)
    cv_results = cross_validate(rf,X,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
    test_rmse2.append( np.mean(np.sqrt(cv_results['test_score']*(-1.))) )
    oob2.append( 1-np.mean([i.oob_score_ for i in cv_results['estimator']]) )
    rf = RandomForestRegressor(n_estimators=i,max_depth=4,bootstrap=True,max_features=3,oob_score=True,warm_start=True)
    cv_results = cross_validate(rf,X,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
    test_rmse3.append( np.mean(np.sqrt(cv_results['test_score']*(-1.))) )
    oob3.append( 1-np.mean([i.oob_score_ for i in cv_results['estimator']]) )
    rf = RandomForestRegressor(n_estimators=i,max_depth=4,bootstrap=True,max_features=4,oob_score=True,warm_start=True)
    cv_results = cross_validate(rf,X,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
    test_rmse4.append( np.mean(np.sqrt(cv_results['test_score']*(-1.))) )
    oob4.append( 1-np.mean([i.oob_score_ for i in cv_results['estimator']]) )
    rf = RandomForestRegressor(n_estimators=i,max_depth=4,bootstrap=True,max_features=5,oob_score=True,warm_start=True)
    cv_results = cross_validate(rf,X,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
    test_rmse5.append( np.mean(np.sqrt(cv_results['test_score']*(-1.))) )
    oob5.append( 1-np.mean([i.oob_score_ for i in cv_results['estimator']]) )

fig, ax = plt.subplots()
ax.plot(trees,test_rmse1, label='max feature = 1')
ax.plot(trees,test_rmse2, label='max feature = 2')
ax.plot(trees,test_rmse3, label='max feature = 3')
ax.plot(trees,test_rmse4, label='max feature = 4')
ax.plot(trees,test_rmse5, label='max feature = 5')
ax.legend(loc='best')
plt.xlabel("Number of trees"); plt.ylabel("Average Test RMSE"); plt.title("Random Forest Regression model")
plt.show()

fig, ax = plt.subplots()
ax.plot(trees,oob1, label='max feature = 1')
ax.plot(trees,oob2, label='max feature = 2')
ax.plot(trees,oob3, label='max feature = 3')
ax.plot(trees,oob4, label='max feature = 4')
ax.plot(trees,oob5, label='max feature = 5')
ax.legend(loc='best')
plt.xlabel("Number of trees"); plt.ylabel("Average Out of Bag Error"); plt.title("Random Forest Regression model")
plt.show()

depth = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30]
test_rmse = []; oob = []
for i in depth:
    print(i)
    rf = RandomForestRegressor(n_estimators=166,max_features=4,max_depth=i,bootstrap=True,oob_score=True,warm_start=True)
    cv_results = cross_validate(rf,X,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
    test_rmse.append( np.mean(np.sqrt(cv_results['test_score']*(-1.))) )
    oob.append( 1-np.mean([i.oob_score_ for i in cv_results['estimator']]) )

fig, ax = plt.subplots()
ax.plot(depth,test_rmse)
ax.legend(loc='best')
plt.xlabel("Max Depth"); plt.ylabel("Average Test RMSE"); plt.title("Random Forest Regression model")
plt.show()

fig, ax = plt.subplots()
ax.plot(depth,oob)
ax.legend(loc='best')
plt.xlabel("Max Depth"); plt.ylabel("Average Out of Bag Error"); plt.title("Random Forest Regression model")
plt.show()

rf = RandomForestRegressor(n_estimators=166,max_features=4,max_depth=10,bootstrap=True,oob_score=True,warm_start=True)
cv_results = cross_validate(rf,X,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
print( np.sqrt(cv_results['test_score']*(-1.)) )
print("Feature importances from best random forest model: \n", cv_results['estimator'][4].feature_importances_)

best_est = cv_results['estimator'][4]
y_pred = best_est.predict(X)
x = np.arange(len(y))+1

fig, ax = plt.subplots()
ax.plot(x,y, 'ro', label='True')
ax.plot(x,y_pred, 'bo', label='Fitted')
ax.legend(loc='best')
plt.xlabel("Data Point"); plt.ylabel("Backup Size (GB)"); plt.title("Random Forest Regression Fitted VS True")
plt.show()

res = y - y_pred
fig, ax = plt.subplots()
ax.plot(x,res, 'ro', label='Residual')
ax.plot(x,y_pred, 'bo', label='Fitted')
ax.legend(loc='best')
plt.xlabel("Data Point"); plt.ylabel("Backup Size (GB)"); plt.title("Random Forest Regression Fitted VS Residuals")
plt.show()

rf = RandomForestRegressor(n_estimators=166,max_features=4,max_depth=4,bootstrap=True,oob_score=True,warm_start=True)
rf.fit(X,y)
tree1 = rf.estimators_[1]
#export_graphviz(tree1,out_file='tree.dot',feature_names=["Week Number", "Day of Week","Backup Start Time","Work Flow","Backup Size"],filled=True,rounded=True)
#from subprocess import check_call
#check_call(['dot','-Tpng','tree.dot','-o','tree.png'])


##### QUESTION 2C
from sklearn.neural_network import MLPRegressor

X = OneHotEncoder().fit_transform(net_back.iloc[:,0:5].values)
y = net_back.iloc[:,5].values

neurons = [2,5,10,50,100,150,200,250,300,350,400,450,500,550,600]
error_relu = []; train_relu = []
error_log = []; train_log = []
error_tanh = []; train_tanh = []
for i in neurons:
    print(i)
    cv_results = cross_validate(MLPRegressor(hidden_layer_sizes=(i,), activation='relu'),X,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
    train_rmse = np.mean(np.sqrt(cv_results['train_score']*(-1.)))
    train_relu.append(train_rmse)
    test_rmse = np.mean(np.sqrt(cv_results['test_score']*(-1.)))
    error_relu.append(test_rmse)
    
for i in neurons:
    print(i)
    cv_results = cross_validate(MLPRegressor(hidden_layer_sizes=(i,), activation='logistic'),X,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
    train_rmse = np.mean(np.sqrt(cv_results['train_score']*(-1.)))
    train_log.append(train_rmse)
    test_rmse = np.mean(np.sqrt(cv_results['test_score']*(-1.)))
    error_log.append(test_rmse)

for i in neurons:
    print(i)
    cv_results = cross_validate(MLPRegressor(hidden_layer_sizes=(i,), activation='tanh'),X,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
    train_rmse = np.mean(np.sqrt(cv_results['train_score']*(-1.)))
    train_tanh.append(train_rmse)
    test_rmse = np.mean(np.sqrt(cv_results['test_score']*(-1.)))
    error_tanh.append(test_rmse)

fig, ax = plt.subplots()
ax.plot(neurons,error_relu, label='ReLU Test RMSE')
ax.plot(neurons,train_relu, label='ReLU Train RMSE')
ax.plot(neurons,error_log, label='Logistic Test RMSE')
ax.plot(neurons,train_log, label='Logistic Train RMSE')
ax.plot(neurons,error_tanh, label='Tanh Test RMSE')
ax.plot(neurons,train_tanh, label='Tanh Train RMSE')
ax.legend(loc='best')
plt.xlabel("Number of neurons"); plt.ylabel("Average RMSE"); plt.title("Neural Network Regression model")
plt.show()

cv_results = cross_validate(MLPRegressor(hidden_layer_sizes=(200,), activation='relu'),X,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
print(np.sqrt(cv_results['test_score']*(-1.)))
best_est = cv_results['estimator'][9]
y_pred = best_est.predict(X)
x = np.arange(len(y))+1

fig, ax = plt.subplots()
ax.plot(x,y, 'ro', label='True')
ax.plot(x,y_pred, 'bo', label='Fitted')
ax.legend(loc='best')
plt.xlabel("Data Point"); plt.ylabel("Backup Size (GB)"); plt.title("Neural Network Regression Fitted VS True")
plt.show()

res = y - y_pred
fig, ax = plt.subplots()
ax.plot(x,res, 'ro', label='Residual')
ax.plot(x,y_pred, 'bo', label='Fitted')
ax.legend(loc='best')
plt.xlabel("Data Point"); plt.ylabel("Backup Size (GB)"); plt.title("Neural Network Regression Fitted VS Residuals")
plt.show()


##### QUESTION 2D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

X_wf0 = []; y_wf0 = []
X_wf1 = []; y_wf1 = []
X_wf2 = []; y_wf2 = []
X_wf3 = []; y_wf3 = []
X_wf4 = []; y_wf4 = []
for n,i in net_back.groupby(net_back["Work-Flow-ID"],sort=False):
    if n == 'work_flow_0':
        X_wf0 = i.drop(columns=['Size of Backup (GB)','Work-Flow-ID'])
        y_wf0 = i["Size of Backup (GB)"].values
    elif n == 'work_flow_1':
        X_wf1 = i.drop(columns=['Size of Backup (GB)','Work-Flow-ID'])
        y_wf1 = i["Size of Backup (GB)"].values
    elif n == 'work_flow_2':
        X_wf2 = i.drop(columns=['Size of Backup (GB)','Work-Flow-ID'])
        y_wf2 = i["Size of Backup (GB)"].values
    elif n == 'work_flow_3':
        X_wf3 = i.drop(columns=['Size of Backup (GB)','Work-Flow-ID'])
        y_wf3 = i["Size of Backup (GB)"].values
    elif n == 'work_flow_4':
        X_wf4 = i.drop(columns=['Size of Backup (GB)','Work-Flow-ID'])
        y_wf4 = i["Size of Backup (GB)"].values

X_wf0 = OrdinalEncoder().fit_transform(X_wf0.iloc[:,0:4].values)
X_wf1 = OrdinalEncoder().fit_transform(X_wf1.iloc[:,0:4].values)
X_wf2 = OrdinalEncoder().fit_transform(X_wf2.iloc[:,0:4].values)
X_wf3 = OrdinalEncoder().fit_transform(X_wf3.iloc[:,0:4].values)
X_wf4 = OrdinalEncoder().fit_transform(X_wf4.iloc[:,0:4].values)

cv_results = cross_validate(LinearRegression(),X_wf0,y_wf0,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
train_rmse = np.mean(np.sqrt(cv_results['train_score']*(-1.))); test_rmse = np.mean(np.sqrt(cv_results['test_score']*(-1.)))
print("Average Train RMSE for Workflow 0 using 10-fold cross validation: ", train_rmse); print("Average Test RMSE for Workflow 0 using 10-fold cross validation: ", test_rmse)

cv_results = cross_validate(LinearRegression(),X_wf1,y_wf1,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
train_rmse = np.mean(np.sqrt(cv_results['train_score']*(-1.))); test_rmse = np.mean(np.sqrt(cv_results['test_score']*(-1.)))
print("Average Train RMSE for Workflow 1 using 10-fold cross validation: ", train_rmse); print("Average Test RMSE for Workflow 1 using 10-fold cross validation: ", test_rmse)

cv_results = cross_validate(LinearRegression(),X_wf2,y_wf2,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
train_rmse = np.mean(np.sqrt(cv_results['train_score']*(-1.))); test_rmse = np.mean(np.sqrt(cv_results['test_score']*(-1.)))
print("Average Train RMSE for Workflow 2 using 10-fold cross validation: ", train_rmse); print("Average Test RMSE for Workflow 2 using 10-fold cross validation: ", test_rmse)

cv_results = cross_validate(LinearRegression(),X_wf3,y_wf3,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
train_rmse = np.mean(np.sqrt(cv_results['train_score']*(-1.))); test_rmse = np.mean(np.sqrt(cv_results['test_score']*(-1.)))
print("Average Train RMSE for Workflow 3 using 10-fold cross validation: ", train_rmse); print("Average Test RMSE for Workflow 3 using 10-fold cross validation: ", test_rmse)

cv_results = cross_validate(LinearRegression(),X_wf4,y_wf4,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
train_rmse = np.mean(np.sqrt(cv_results['train_score']*(-1.))); test_rmse = np.mean(np.sqrt(cv_results['test_score']*(-1.)))
print("Average Train RMSE for Workflow 4 using 10-fold cross validation: ", train_rmse); print("Average Test RMSE for Workflow 4 using 10-fold cross validation: ", test_rmse)

wf0_train = []; wf0_test = []
wf1_train = []; wf1_test = []
wf2_train = []; wf2_test = []
wf3_train = []; wf3_test = []
wf4_train = []; wf4_test = []
degree = np.arange(2,16)
for i in degree:
    print(i)
    model = make_pipeline(PolynomialFeatures(i), LinearRegression())
    cv_results = cross_validate(model,X_wf0,y_wf0,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
    wf0_train.append( np.mean(np.sqrt(cv_results['train_score']*(-1.))) )
    wf0_test.append( np.mean(np.sqrt(cv_results['test_score']*(-1.))) )
    cv_results = cross_validate(model,X_wf1,y_wf1,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
    wf1_train.append( np.mean(np.sqrt(cv_results['train_score']*(-1.))) )
    wf1_test.append( np.mean(np.sqrt(cv_results['test_score']*(-1.))) )
    cv_results = cross_validate(model,X_wf2,y_wf2,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
    wf2_train.append( np.mean(np.sqrt(cv_results['train_score']*(-1.))) )
    wf2_test.append( np.mean(np.sqrt(cv_results['test_score']*(-1.))) )
    cv_results = cross_validate(model,X_wf3,y_wf3,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
    wf3_train.append( np.mean(np.sqrt(cv_results['train_score']*(-1.))) )
    wf3_test.append( np.mean(np.sqrt(cv_results['test_score']*(-1.))) )
    cv_results = cross_validate(model,X_wf4,y_wf4,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
    wf4_train.append( np.mean(np.sqrt(cv_results['train_score']*(-1.))) )
    wf4_test.append( np.mean(np.sqrt(cv_results['test_score']*(-1.))) )

fig, ax = plt.subplots()
ax.plot(degree,wf0_train, label='work_flow_0')
ax.plot(degree,wf1_train, label='work_flow_1')
ax.plot(degree,wf2_train, label='work_flow_2')
ax.plot(degree,wf3_train, label='work_flow_3')
ax.plot(degree,wf4_train, label='work_flow_4')
ax.legend(loc='best')
plt.xlabel("Degree of Polynomial"); plt.ylabel("Average Train RMSE"); plt.title("Polynomial Regression model")
plt.show()

fig, ax = plt.subplots()
ax.plot(degree,wf0_test, label='work_flow_0')
ax.plot(degree,wf1_test, label='work_flow_1')
ax.plot(degree,wf2_test, label='work_flow_2')
ax.plot(degree,wf3_test, label='work_flow_3')
ax.plot(degree,wf4_test, label='work_flow_4')
ax.legend(loc='best')
plt.xlabel("Degree of Polynomial"); plt.ylabel("Average Test RMSE"); plt.title("Polynomial Regression model")
plt.show()

model = make_pipeline(PolynomialFeatures(10), LinearRegression())
cv_results = cross_validate(model,X_wf4,y_wf4,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
print(np.sqrt(cv_results['test_score']*(-1.)))

best_est = cv_results['estimator'][4]
y_pred = best_est.predict(X_wf4)
x = np.arange(len(y_wf4))+1

fig, ax = plt.subplots()
ax.plot(x,y_wf4, 'ro', label='True')
ax.plot(x,y_pred, 'bo', label='Fitted')
ax.legend(loc='best')
plt.xlabel("Data Point"); plt.ylabel("Backup Size (GB)"); plt.title("Polynomial features(workflow_4) Regression Fitted VS True")
plt.show()

res = y_wf4 - y_pred
fig, ax = plt.subplots()
ax.plot(x,res, 'ro', label='Residual')
ax.plot(x,y_pred, 'bo', label='Fitted')
ax.legend(loc='best')
plt.xlabel("Data Point"); plt.ylabel("Backup Size (GB)"); plt.title("Polynomial features(workflow_4) Regression Fitted VS Residuals")
plt.show()


##### QUESTION 2E
from sklearn.neighbors import KNeighborsRegressor

X = OrdinalEncoder().fit_transform(net_back.iloc[:,0:5].values)
y = net_back.iloc[:,5].values
neighbors = [2,3,5,10,15,20,25,30,40,50,100,150,200,250,300,350,400,450,500]
error = []
for i in neighbors:
    print(i)
    cv_results = cross_validate(KNeighborsRegressor(n_neighbors=i),X,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
    test_rmse = np.mean(np.sqrt(cv_results['test_score']*(-1.)))
    error.append(test_rmse)

fig, ax = plt.subplots()
ax.plot(neighbors,error, 'r')
ax.legend(loc='best')
plt.xlabel("Number of neighbors"); plt.ylabel("Average Test RMSE"); plt.title("k-NN Regression model")
plt.show()

cv_results = cross_validate(KNeighborsRegressor(n_neighbors=3),X,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
best_est = cv_results['estimator'][1]
y_pred = best_est.predict(X)
x = np.arange(len(y))+1

fig, ax = plt.subplots()
ax.plot(x,y, 'ro', label='True')
ax.plot(x,y_pred, 'bo', label='Fitted')
ax.legend(loc='best')
plt.xlabel("Data Point"); plt.ylabel("Backup Size (GB)"); plt.title("k-NN Regression Fitted VS True")
plt.show()

res = y - y_pred
fig, ax = plt.subplots()
ax.plot(x,res, 'ro', label='Residual')
ax.plot(x,y_pred, 'bo', label='Fitted')
ax.legend(loc='best')
plt.xlabel("Data Point"); plt.ylabel("Backup Size (GB)"); plt.title("k-NN Regression Fitted VS Residuals")
plt.show()


############################################################################### DATASET 2 ############################################################################################

##### QUESTION 2
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.feature_selection import f_regression

housing = pd.read_csv("housing_data.csv")
X = housing.iloc[:,0:13].values
y = housing.iloc[:,13].values

F, pval = f_regression(X,y)
print("F-score of variables: \n", F)
print("p-value of variables: \n", pval)

cv_results = cross_validate(LinearRegression(),X,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
train_rmse = np.sqrt(cv_results['train_score']*(-1.))
test_rmse = np.sqrt(cv_results['test_score']*(-1.))
print("Train RMSE for Linear Regression for each fold in 10-fold cross validation: \n", train_rmse)
print("Average Train RMSE for Linear Regression: ", np.mean(train_rmse))
print("Test RMSE for Linear Regression for each fold in 10-fold cross validation: \n", test_rmse)
print("Average Test RMSE for Linear Regression: ", np.mean(test_rmse))

best_est = cv_results['estimator'][0]
y_pred = best_est.predict(X)
x = np.arange(len(y))+1
best_est_coef = best_est.coef_
print("Coefficients of best unregularized estimator: \n", best_est_coef)

fig, ax = plt.subplots()
ax.plot(x,y, 'ro', label='True')
ax.plot(x,y_pred, 'bo', label='Fitted')
ax.legend(loc='best')
plt.xlabel("Data Point"); plt.ylabel("MEDV"); plt.title("Fitted VS True")
plt.show()

res = y - y_pred
fig, ax = plt.subplots()
ax.plot(x,res, 'ro', label='Residual')
ax.plot(x,y_pred, 'bo', label='Fitted')
ax.legend(loc='best')
plt.xlabel("Data Point"); plt.ylabel("MEDV"); plt.title("Fitted VS Residuals")
plt.show()


##### QUESTION 3

housing = pd.read_csv("housing_data.csv")
X = housing.iloc[:,0:13].values
y = housing.iloc[:,13].values

alpha = [0.0001,0.001,0.01,0.1,0.2,0.3,0.5,0.6,0.7,0.9,1,10,100,1000,10000]
cv_results = {}
for i in alpha:
    cv_results = cross_validate(Lasso(alpha=i,max_iter=1000),X,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
    train_rmse = np.mean(np.sqrt(cv_results['train_score']*(-1.)))
    test_rmse = np.mean(np.sqrt(cv_results['test_score']*(-1.)))
    print("Average Train RMSE for Lasso Regularizer with alpha = %s : %.6f" %(i,train_rmse))

for i in alpha:
    cv_results = cross_validate(Lasso(alpha=i,max_iter=1000),X,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
    train_rmse = np.mean(np.sqrt(cv_results['train_score']*(-1.)))
    test_rmse = np.mean(np.sqrt(cv_results['test_score']*(-1.)))
    print("Average Test RMSE for Lasso Regularizer with alpha = %s : %.6f" %(i,test_rmse))

alpha1 = [0.001,0.01,0.1,1,10]
alpha2 = [0.001,0.01,0.1,1,10]
import warnings
warnings.filterwarnings("ignore")

cv_results = {}
for i in alpha1:
    for j in alpha2:
        cv_results = cross_validate(ElasticNet(alpha=i,l1_ratio=j,max_iter=1000),X,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
        train_rmse = np.mean(np.sqrt(cv_results['train_score']*(-1.)))
        test_rmse = np.mean(np.sqrt(cv_results['test_score']*(-1.)))
        print("Average Train RMSE for Elastic Net Regularizer with lambda1 = %s and lambda2 = %s: %.6f" %(i,j,test_rmse))

for i in alpha1:
    for j in alpha2:
        cv_results = cross_validate(ElasticNet(alpha=i,l1_ratio=j,max_iter=1000),X,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
        train_rmse = np.mean(np.sqrt(cv_results['train_score']*(-1.)))
        test_rmse = np.mean(np.sqrt(cv_results['test_score']*(-1.)))
        print("Average Test RMSE for Elastic Net Regularizer with lambda1 = %s and lambda2 = %s: %.6f" %(i,j,test_rmse))

best_ridge_est = Ridge(alpha=100)
ridge_coef = best_ridge_est.fit(X,y).coef_
print("Coefficients for best Ridge Regularizer: \n", ridge_coef)

best_lasso_est = Lasso(alpha=0.1)
lasso_coef = best_lasso_est.fit(X,y).coef_
print("Coefficients for best Lasso Regularizer: \n", lasso_coef)

best_elastic_est = ElasticNet(alpha=0.1,l1_ratio=0.001)
elastic_coef = best_elastic_est.fit(X,y).coef_
print("Coefficients for best Elastic Net Regularizer: \n", elastic_coef)

x = np.arange(1,14)
fig, ax = plt.subplots()
ax.plot(x,elastic_coef, label='Elastic Net')
ax.plot(x,ridge_coef, label='Ridge')
ax.plot(x,lasso_coef, label='Lasso')
ax.plot(x,best_est_coef, label='Unregularized')
ax.legend(loc='best')
plt.xlabel("Coefficient Index"); plt.ylabel("Magnitude"); plt.title("Linear Regression Coefficients")
plt.show()


############################################################################### DATASET 3 ############################################################################################

##### QUESTION 1
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer, make_column_transformer

insurance = pd.read_csv("insurance_data.csv")
X = insurance.iloc[:,0:6].values
y = insurance.iloc[:,6].values

ft0 = ['ft1','ft2','ft3','ft4', 'ft5', 'ft6']
preprocess = make_column_transformer((OneHotEncoder(sparse=False), ['ft4','ft5','ft6']),remainder='passthrough')
X_feat = preprocess.fit_transform(insurance[ft0])

cv_results = cross_validate(LinearRegression(),X_feat,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
train_rmse = np.sqrt(cv_results['train_score']*(-1.))
test_rmse = np.sqrt(cv_results['test_score']*(-1.))
print("Train RMSE for Linear Regression with Feature Encoding for each fold in 10-fold cross validation: \n", train_rmse)
print("Average Train RMSE for Linear Regression with Feature Encoding: ", np.mean(train_rmse))
print("Test RMSE for Linear Regression with Feature Encoding for each fold in 10-fold cross validation: \n", test_rmse)
print("Average Test RMSE for Linear Regression with Feature Encoding: ", np.mean(test_rmse))

best_est = cv_results['estimator'][5]
y_pred = best_est.predict(X_feat)
x = np.arange(len(y))+1

fig, ax = plt.subplots()
ax.plot(x,y, 'ro', label='True')
ax.plot(x,y_pred, 'bo', label='Fitted')
ax.legend(loc='best')
plt.xlabel("Data Point"); plt.ylabel("Charges"); plt.title("Feature Encoding Fitted VS True")
plt.show()

res = y - y_pred
fig, ax = plt.subplots()
ax.plot(x,res, 'ro', label='Residual')
ax.plot(x,y_pred, 'bo', label='Fitted')
ax.legend(loc='best')
plt.xlabel("Data Point"); plt.ylabel("Charges"); plt.title("Feature Encoding Fitted VS Residuals")
plt.show()

ft0 = ['ft1','ft2','ft3','ft4', 'ft5', 'ft6']
preprocess = make_column_transformer((StandardScaler(),['ft1', 'ft2','ft3']),(OneHotEncoder(sparse=False), ['ft4','ft5','ft6']))
X_stand = preprocess.fit_transform(insurance[ft0])

cv_results = cross_validate(LinearRegression(),X_stand,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
train_rmse = np.sqrt(cv_results['train_score']*(-1.))
test_rmse = np.sqrt(cv_results['test_score']*(-1.))
print("Train RMSE for Linear Regression with Standardization for each fold in 10-fold cross validation: \n", train_rmse)
print("Average Train RMSE for Linear Regression with Standardization: ", np.mean(train_rmse))
print("Test RMSE for Linear Regression with Standardization for each fold in 10-fold cross validation: \n", test_rmse)
print("Average Test RMSE for Linear Regression with Standardization: ", np.mean(test_rmse))

best_est = cv_results['estimator'][5]
y_pred = best_est.predict(X_stand)
x = np.arange(len(y))+1

fig, ax = plt.subplots()
ax.plot(x,y, 'ro', label='True')
ax.plot(x,y_pred, 'bo', label='Fitted')
ax.legend(loc='best')
plt.xlabel("Data Point"); plt.ylabel("Charges"); plt.title("Standardization Fitted VS True")
plt.show()

res = y - y_pred
fig, ax = plt.subplots()
ax.plot(x,res, 'ro', label='Residual')
ax.plot(x,y_pred, 'bo', label='Fitted')
ax.legend(loc='best')
plt.xlabel("Data Point"); plt.ylabel("Charges"); plt.title("Standardization Fitted VS Residuals")
plt.show()

for n,i in enumerate(X[:,0]):
    if i < 30:
        X[:,0][n] = 1
    elif i > 50:
        X[:,0][n] = 3
    else:
        X[:,0][n] = 2
        
X[:,1:3] = StandardScaler().fit_transform(X[:,1:3])
X_out = OneHotEncoder(sparse=False).fit_transform(X[:,3:6])
X = X[:,0:3]
X = np.column_stack((X,X_out))

cv_results = cross_validate(LinearRegression(),X,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
train_rmse = np.sqrt(cv_results['train_score']*(-1.))
test_rmse = np.sqrt(cv_results['test_score']*(-1.))
print("Train RMSE for Linear Regression with Dividing into 3 Ranges for each fold in 10-fold cross validation: \n", train_rmse)
print("Average Train RMSE for Linear Regression with Dividing into 3 Ranges: ", np.mean(train_rmse))
print("Test RMSE for Linear Regression with Dividing into 3 Ranges for each fold in 10-fold cross validation: \n", test_rmse)
print("Average Test RMSE for Linear Regression with Dividing into 3 Ranges: ", np.mean(test_rmse))

best_est = cv_results['estimator'][5]
y_pred = best_est.predict(X)
x = np.arange(len(y))+1

fig, ax = plt.subplots()
ax.plot(x,y, 'ro', label='True')
ax.plot(x,y_pred, 'bo', label='Fitted')
ax.legend(loc='best')
plt.xlabel("Data Point"); plt.ylabel("Charges"); plt.title("Dividing into 3 Ranges Fitted VS True")
plt.show()

res = y - y_pred
fig, ax = plt.subplots()
ax.plot(x,res, 'ro', label='Residual')
ax.plot(x,y_pred, 'bo', label='Fitted')
ax.legend(loc='best')
plt.xlabel("Data Point"); plt.ylabel("Charges"); plt.title("Dividing into 3 Ranges Fitted VS Residuals")
plt.show()


##### QUESTION 2
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.preprocessing import OrdinalEncoder

insurance = pd.read_csv("insurance_data.csv")
X = insurance.iloc[:,0:6].values
y = insurance.iloc[:,6].values

X[:,3:] = OrdinalEncoder().fit_transform(X[:,3:])
F, pval = f_regression(X,y)
print("F-score of variables using f_regression: \n", F)
print("p-value of variables using f_regression: \n", pval)

mi = mutual_info_regression(X,y)
print("Mutual information between variables and target using mutual_info_regression: \n", mi)

X = insurance.iloc[:,0:6].values
ft1 = X[:,0]
ft2 = X[:,1]
color = []
for i in X[:,4]:
    if i == 'yes':
        color.append('red')
    else:
        color.append('black')

plt.scatter(ft1,y,c=color)
plt.xlabel("ft1"); plt.ylabel("Charges"); plt.title("Charges VS ft2 Correlation")
plt.show()

##### QUESTION 3

insurance = pd.read_csv("insurance_data.csv")
X = insurance.iloc[:,0:6].values
y = insurance.iloc[:,6].values

ft0 = ['ft1','ft2','ft3','ft4', 'ft5', 'ft6']
preprocess = make_column_transformer((StandardScaler(),['ft1', 'ft2','ft3']),(OneHotEncoder(sparse=False), ['ft4','ft5','ft6']))
X_stand = preprocess.fit_transform(insurance[ft0])
log_y = np.log(insurance.iloc[:,6].values)

cv_results = cross_validate(LinearRegression(),X_stand,log_y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
train_rmse = np.sqrt(cv_results['train_score']*(-1.))
test_rmse = np.sqrt(cv_results['test_score']*(-1.))
print("Train RMSE for Linear Regression with modified Target Variable for each fold in 10-fold cross validation: \n", train_rmse)
print("Average Train RMSE for Linear Regression with modified Target Variable: ", np.mean(train_rmse))
print("Test RMSE for Linear Regression with modified Target Variable for each fold in 10-fold cross validation: \n", test_rmse)
print("Average Test RMSE for Linear Regression with modified Target Variable: ", np.mean(test_rmse))

best_est = cv_results['estimator'][5]
y_pred = np.exp(best_est.predict(X_stand))
x = np.arange(len(y))+1

fig, ax = plt.subplots()
ax.plot(x,y, 'ro', label='True')
ax.plot(x,y_pred, 'bo', label='Fitted')
ax.legend(loc='best')
plt.xlabel("Data Point"); plt.ylabel("Charges"); plt.title("Modified Target Variable Fitted VS True")
plt.show()

res = y - y_pred
fig, ax = plt.subplots()
ax.plot(x,res, 'ro', label='Residual')
ax.plot(x,y_pred, 'bo', label='Fitted')
ax.legend(loc='best')
plt.xlabel("Data Point"); plt.ylabel("Charges"); plt.title("Modified Target Variable Fitted VS Residuals")
plt.show()

F, pval = f_regression(X_stand,log_y)
print("F-score of variables using f_regression: \n", F)
print("p-value of variables using f_regression: \n", pval)
mi = mutual_info_regression(X_stand,log_y)
print("Mutual information between variables and target using mutual_info_regression: \n", mi)

ft1 = X[:,0]
ft2 = X[:,1]
color = []
for i in X[:,4]:
    if i == 'yes':
        color.append('red')
    else:
        color.append('black')

plt.scatter(ft1,log_y,c=color)
plt.xlabel("ft1"); plt.ylabel("Charges"); plt.title("Charges VS ft1 Correlation")
plt.show()


##### QUESTION 4
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

insurance = pd.read_csv("insurance_data.csv")
X = insurance.iloc[:,0:6].values
y = insurance.iloc[:,6].values

ft0 = ['ft1','ft2','ft3','ft4', 'ft5', 'ft6']
preprocess = make_column_transformer((StandardScaler(),['ft1', 'ft2','ft3']),(OneHotEncoder(sparse=False), ['ft4','ft5','ft6']))
X_stand = preprocess.fit_transform(insurance[ft0])

degree = np.arange(1,6,dtype=int)
train, test = [],[]
for i in degree:
    print(i)
    model = make_pipeline(PolynomialFeatures(i), LinearRegression())
    cv_results = cross_validate(model,X_stand,y,scoring='neg_mean_squared_error',cv=10,return_train_score=True,return_estimator=True)
    train.append( np.mean(np.sqrt(cv_results['train_score']*(-1.))) )
    test.append( np.mean(np.sqrt(cv_results['test_score']*(-1.))) )

fig, ax = plt.subplots()
ax.plot(degree,train, label='Train')
ax.plot(degree,test, label='Test')
ax.legend(loc='best')
plt.xlabel("Degree of Polynomial"); plt.ylabel("Average RMSE"); plt.title("Polynomial features")
plt.tight_layout()
plt.show()

trees = np.arange(4,204,4,dtype=int); depth = np.arange(2,16,dtype=int)
param = {'n_estimators':trees,'max_depth':depth}
rf = RandomForestRegressor()
clf = GridSearchCV(rf,param,cv=10)
clf.fit(X_stand,y)
print("Best paramters of Random Forest model: \n", clf.best_params_)
print("Best cross-validated score of Random Forest model: ", clf.best_score_)


neurons = [2,5,10,50,100,150,200,250,300,350,400,450,500,550,600]
act = ['relu','tanh','logistic']
param = {'hidden_layer_sizes':neurons,'activation':act}
nn = MLPRegressor(warm_start=True,early_stopping=True,learning_rate='adaptive',max_iter=1000)
clf = GridSearchCV(nn,param,cv=10)
clf.fit(X_stand,y)
print("Best paramters of Neural Network model: \n", clf.best_params_)
print("Best cross-validated score of Neural Network model: ", clf.best_score_)


trees = np.arange(4,204,4,dtype=int); depth = np.arange(2,16,dtype=int)
param = {'n_estimators':trees,'max_depth':depth}
gb = GradientBoostingRegressor()
clf = GridSearchCV(gb,param,cv=10)
clf.fit(X_stand,y)
print("Best paramters of Gradient Boosting model: \n", clf.best_params_)
print("Best cross-validated score of Gradient Boosting model: ", clf.best_score_)


