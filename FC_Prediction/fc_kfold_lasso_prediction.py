#coding: utf-8
import os
import random
import numpy as np
import nibabel as nib
import glob
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.ensemble import BaggingRegressor
import joblib
from sklearn.metrics import r2_score, make_scorer
from datetime import datetime
from sklearn.svm import LinearSVR
import ToolBox as tb
import sys
import math
from sklearn import linear_model
sys.path.append('/home/cuizaixu_lab/fanqingchen/DATA/Code/PLSR_Prediction')
from step1_PLSr.step1_PLSr import Setparameter

def my_scorer(y_true, y_predicted):
    mae = np.mean(np.abs(y_true - y_predicted))

    Predict_Score_new = np.transpose(y_predicted)
    Corr = np.corrcoef(Predict_Score_new, y_true)
    Corr = Corr[0, 1]

    error = (1/mae)+Corr
    return error
def LoadData(datapath, labelpath, dimention, Time,Permutation=0):

    data_list = []
    #Loading Data
    #data_files_all = sorted(glob.glob(datapath), reverse=True)
    data_files_all = np.loadtxt(datapath)
    label_files_all = pd.read_csv(labelpath)
    label = label_files_all[dimention]

    #Label
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
    # if do permutation , random data
    if Permutation:
        np.random.shuffle(x_data)


    data_list.append(x_data)
    data_list.append(y_label)
    return data_list

def PLSPrediction_Model(data_list, dimention, weightpath, Permutation,kfold, Time=1):
    epoch = 0
    count = tb.countnum()
    outer_results_R = []
    outer_results_mae = []
    outer_results_r2 = []
    feature_weight_res = np.zeros([61776, 1])

    x_data = data_list[0]
    y_label = data_list[1]
    kf = KFold(n_splits=kfold, shuffle=True)
    for train_index, test_index in kf.split(x_data):
        epoch = epoch + 1
        # split data
        X_train, X_test = x_data[train_index, :], x_data[test_index, :]
        y_train, y_test = y_label[train_index], y_label[test_index]

        normalize = preprocessing.MinMaxScaler()
        Subjects_Data_train = normalize.fit_transform(X_train)
        Subjects_Data_test = normalize.transform(X_test)

        #tb.ToolboxCSV_server('train_set_bagging_' + dimention + '_' + str(Time) + '_' + str(epoch) + '.csv', X_train)
        #tb.ToolboxCSV_server('train_label_bagging_' + dimention + '_' + str(Time) + '_' + str(epoch) + '.csv', y_train)
        #tb.ToolboxCSV_server('test_set_bagging_' + dimention + '_' + str(Time) + '_' + str(epoch) + '.csv', X_test)
        if Permutation == 0:
          tb.ToolboxCSV_server('test_label_bagging_' + dimention + '_' + str(Time) + '_'+str(count) + '_' + str(epoch) + '.csv', y_test)

        # Model
        lasso = linear_model.Lasso()
        param_grid = {
            'alpha': [0.1, 0.2, 0.3]
        }
        cv_time = 5
        predict_model = GridSearchCV(lasso, param_grid=param_grid, cv=cv_time)
        best_model = predict_model.best_estimator_
        best_model.fit(Subjects_Data_train, y_train)

        # weight
        feature_weight = best_model.coef_
        feature_weight_res = np.add(feature_weight_res, feature_weight)
        Predict_Score = best_model.predict(Subjects_Data_test)

        tb.ToolboxCSV_server('Predict_Score_bagging_' + dimention + '_' + str(Time) + '_' + str(count)+ '_' + str(epoch) + '.csv',
                             Predict_Score)

        Predict_Score_new = np.transpose(Predict_Score)
        Corr = np.corrcoef(Predict_Score_new, y_test)
        Corr = Corr[0, 1]
        outer_results_R.append(Corr)

        MAE_inv = round(np.mean(np.abs(Predict_Score - y_test)), 4)
        outer_results_mae.append(MAE_inv)

        r2 = r2_score(y_test, Predict_Score_new)
        outer_results_r2.append(r2)

       # print('>Corr=%.3f, MAE=%.3f, r2=%.3f,est=%.3f, cfg=%s' % (Corr, MAE_inv, r2, predict_model.best_score_, predict_model.best_params_))
    feature_weight_res_mean = feature_weight_res / kfold
    feature_weight_file = pd.DataFrame(feature_weight_res_mean)
    if Permutation:
       feature_weight_file.to_csv(weightpath + 'pt/feature_weight_' + str(round(np.mean(outer_results_R), 3)) + '_'+str(count) + '_' +dimention + '.csv')
    else:
       feature_weight_file.to_csv(weightpath + 'tw/feature_weight_' + str(round(np.mean(outer_results_R), 3)) + '_'+str(count) + '_' + dimention + '.csv')

    print('Result: R=%.3f ,MAE=%.3f, r2=%.3f' % (np.mean(outer_results_R), np.mean(outer_results_mae), np.mean(outer_results_r2)))

parameter = Setparameter()
data_list = LoadData(
    parameter['datapath'],
    parameter['labelpath'],
    parameter['dimention'],
    parameter['Time'],
    parameter['Permutation']
)
PLSPrediction_Model(data_list, parameter['dimention'], parameter['weightpath'], parameter['Permutation'], parameter['KFold'], parameter['Time'])

