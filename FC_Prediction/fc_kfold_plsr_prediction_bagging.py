#coding: utf-8
import os
import random
import numpy as np
import nibabel as nib
import glob
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.ensemble import BaggingRegressor
import joblib
from sklearn.metrics import r2_score,make_scorer
from datetime import datetime
import ToolBox as tb
import sys
import time
import statsmodels.formula.api as sm
import pingouin as pg
sys.path.append('/home/cuizaixu_lab/fanqingchen/DATA/Code/PLSR_Prediction')
#from step1_PLSr.step1_PLSr import Setparameter
from step1_PLSr.step1_PLSr_HCP import Setparameter  # TODO: 方便我自己，应该是只改step1中的路径就可以了
def my_scorer(y_true, y_predicted):
    mae = np.mean(np.abs(y_true - y_predicted))

    Predict_Score_new = np.transpose(y_predicted)
    Corr = np.corrcoef(Predict_Score_new, y_true)
    Corr = Corr[0, 1]

    error = (1/mae)+Corr
    return error

def LoadData(datapath, labelpath, dimention, covariatespath, Time, Permutation=0):

    data_list = []
    # Loading Data
    # data_files_all = sorted(glob.glob(datapath), reverse=True)
    data_files_all = np.loadtxt(datapath)

    # Label
    label_files_all = pd.read_csv(labelpath)
    label = label_files_all[dimention]
    y_label = np.array(label)

    # #Data
    # files_data = []
    # for i in data_files_all:
    #     img_data = nib.load(i)
    #     img_data = img_data.get_data()
    #     img_data_reshape = tb.upper_tri_indexing(img_data)
    #     files_data.append(img_data_reshape)
    # x_data = np.array(files_data)
    x_data = data_files_all
    print('--x-data--', x_data)
    #Covariates
    Covariates = pd.read_csv(covariatespath)
    Covariates = np.array(Covariates)
    Covariates = Covariates[:, 1:].astype(float)
    # if do permutation , random data
    if Permutation:
        np.random.shuffle(x_data)

    data_list.append(x_data)
    data_list.append(y_label)
    data_list.append(Covariates)
    return data_list

def PLSPrediction_Model(data_list, dimention, weightpath, Permutation, kfold, datamark, Time=1):
    epoch = 0
    count = tb.countnum()
    print('--count--', count)
    outer_results_parR = []
    outer_results_R = []
    outer_results_mae = []
    outer_results_r2 = []

    dataMark = datamark
    x_data = data_list[0]
    y_label = data_list[1]
    Covariates = data_list[2]
    feature_weight_res = np.zeros([np.shape(x_data)[1], 1])
    kf = KFold(n_splits=kfold, shuffle=True)
    for train_index, test_index in kf.split(x_data):
        epoch = epoch + 1
        # split data
        X_train, X_test = x_data[train_index, :], x_data[test_index, :]
        y_train, y_test = y_label[train_index], y_label[test_index]
        Covariates_train, Covariates_test = Covariates[train_index, :], Covariates[test_index, :]

        # # Controlling covariates
        # X_train, X_test = Controllingcovariates(Covariates, X_train, X_test, Covariates_train, Covariates_test)

        normalize = preprocessing.MinMaxScaler()
        Subjects_Data_train = normalize.fit_transform(X_train)
        Subjects_Data_test = normalize.transform(X_test)

        #tb.ToolboxCSV_server(X_train, dataMark, 'train_set_bagging_' + dimention + '_' + str(Time) + '_' + str(count) + '_' + str(epoch) + '.csv')
        tb.ToolboxCSV_server(y_train, dataMark, 'train_label_bagging_' + dimention + '_' + str(Time) + '_' + str(count)+'_' + str(epoch) + '.csv')
        #tb.ToolboxCSV_server(X_test, dataMark, 'test_set_bagging_' + dimention + '_' + str(Time) + '_' + str(count) + '_' + str(epoch) + '.csv')
        if Permutation == 0:
          tb.ToolboxCSV_server(y_test, dataMark, 'test_label_bagging_' + dimention + '_' + str(Time) + '_'+str(count) + '_' + str(epoch) + '.csv')


        # Model
        # bagging,PLS
        bagging = BaggingRegressor(base_estimator=PLSRegression())
#chunhui!!!shuyinuo
        # 网格交叉验证
        my_func = make_scorer(my_scorer, greater_is_better=True)
        cv_times = 2  # inner
        param_grid = {
            'base_estimator__n_components': [1, 2, 3, 4, 5],
            'n_estimators': [10]
        }
        predict_model = GridSearchCV(bagging, param_grid, scoring=my_func, verbose=6, cv=cv_times)

        predict_model.fit(Subjects_Data_train, y_train)
        best_model = predict_model.best_estimator_
        #weight
        feature_weight = np.zeros([np.shape(X_train)[1], 1])
        for i, j in enumerate(predict_model.best_estimator_.estimators_):
            #print('第{}个模型的系数{}'.format(i, j.coef_))
            # test.to_csv('test_'+str(epoch)+'_'+str(i)+'.csv')
            feature_weight = np.add(j.coef_, feature_weight)

        num = len(predict_model.best_estimator_.estimators_)
        feature_weight_mean = feature_weight / num
        print('--feature_weight_mean--\n', feature_weight_mean)
        feature_weight_res = np.add(feature_weight_mean, feature_weight_res)  # sum = sum + 1


        Predict_Score = best_model.predict(Subjects_Data_test)
        tb.ToolboxCSV_server(Predict_Score, dataMark, 'Predict_Score_bagging_' + dimention + '_' + str(Time) + '_' + str(count) + '_' + str(epoch) + '.csv',
                             )
        # Controlling covariates and save parCorr
        preTrueCovari = {"predict": Predict_Score.flatten(),
                         "true": y_test,
                         "age": Covariates_test[:, 0],
                         "sex": Covariates_test[:, 1],
                         "fd": Covariates_test[:, 2]}

        preTrueCovari = pd.DataFrame(preTrueCovari)
        resCovari = pg.partial_corr(data=preTrueCovari, x='true', y='predict', covar=['age', 'sex', 'fd'])
        parCorr = resCovari['r']
        outer_results_parR.append(parCorr)
        # Don't Controlling covariates and save Corr
        Predict_Score_new = np.transpose(Predict_Score)
        Corr = np.corrcoef(Predict_Score_new, y_test)
        Corr = Corr[0, 1]
        outer_results_R.append(Corr)

        MAE_inv = round(np.mean(np.abs(Predict_Score - y_test)), 4)
        outer_results_mae.append(MAE_inv)

        r2 = r2_score(y_test, Predict_Score_new[0])
        outer_results_r2.append(r2)

        print('>parCorr=%.3f,Corr=%.3f, MAE=%.3f, r2=%.3f,est=%.3f, cfg=%s' % (parCorr,Corr, MAE_inv, r2, predict_model.best_score_, predict_model.best_params_))
    feature_weight_res_mean = feature_weight_res / kfold
    feature_weight_file = pd.DataFrame(feature_weight_res_mean)

    if Permutation:

       wpath = weightpath + 'pt/'+str(datetime.now().strftime('%Y_%m_%d'))+'_' + str(Time)
       if not os.path.exists(wpath):
           os.makedirs(wpath)
       feature_weight_file.to_csv(weightpath + 'pt/'+str(datetime.now().strftime('%Y_%m_%d'))+'_'+str(Time)+'/feature_weight_' + str(round(np.mean(outer_results_R), 3)) + '_'+str(round(np.mean(outer_results_parR), 3))+
                                  '_'+str(count) + '_' +dimention + '.csv')
    else:
        wpath = weightpath + 'tw/' + str(datetime.now().strftime('%Y_%m_%d')) + '_' + str(Time)
        if not os.path.exists(wpath):
            os.makedirs(wpath)
        feature_weight_file.to_csv(weightpath + 'tw/' + str(datetime.now().strftime('%Y_%m_%d')) + '_' + str(Time) + '/feature_weight_' + str(round(np.mean(outer_results_R), 3)) + '_' + str(round(np.mean(outer_results_parR), 3)) +
                                   '_' + str(count) + '_' + dimention + '.csv')
    print('Result: Covariates-R=%.3f, R=%.3f ,MAE=%.3f, r2=%.3f' % (np.mean(outer_results_parR), np.mean(outer_results_R), np.mean(outer_results_mae), np.mean(outer_results_r2)))

def Controllingcovariates(Covariates, X_train, X_test, Covariates_train, Covariates_test):
    Features_Quantity = np.shape(X_train)[1]
    Covariates_Quantity = np.shape(Covariates)[1] - 1  # Covariates_Quantity = 4 因为有一列subjectkey
    # Controlling covariates from brain data
    df = {}
    for k in np.arange(Covariates_Quantity):
        df['Covariate_' + str(k)] = Covariates_train[:, k + 1]  # k+1 避免取到第一列subjectkey

    # Construct formula
    Formula = 'Data ~ Covariate_0'
    for k in np.arange(Covariates_Quantity - 1) + 1:
        Formula = Formula + ' + Covariate_' + str(k)  # Formula = Covariate_0 + Covariate_1 + Covariate_2

    # Regress covariates from each brain features
    for k in np.arange(Features_Quantity):
        df['Data'] = X_train[:, k]  # 训练集
        # Regressing covariates using training data
        LinModel_Res = sm.ols(formula=Formula,
                              data=df).fit()  # df{'Data':Subjects_Data_train，'Covariate_0':age,'Covariate_1':sex,'Covariate_2':FD}
        # Using residuals replace the training data
        X_train[:, k] = LinModel_Res.resid  # 回归后的结果(新特征，残差)
        # Calculating the residuals of testing data by applying the coeffcients of training data
        Coefficients = LinModel_Res.params
        X_test[:, k] = X_test[:, k] - Coefficients[0]
        for m in np.arange(Covariates_Quantity):  # [0, 1 , 2 ]
            X_test[:, k] = X_test[:, k] - Coefficients[m + 1] * Covariates_test[:, m + 1]
    return X_train, X_test

parameter = Setparameter()
data_list = LoadData(
    parameter['datapath'],
    parameter['labelpath'],
    parameter['dimention'],
    parameter['covariatespath'],
    parameter['Time'],
    parameter['Permutation']
)

PLSPrediction_Model(data_list, parameter['dimention'], parameter['weightpath'], parameter['Permutation'], parameter['KFold'], parameter['dataMark'], parameter['Time'])

