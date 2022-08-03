#coding: utf-8
import os
import random
import numpy as np
import nibabel as nib
import glob
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingRegressor
import joblib
from datetime import datetime
import ToolBox as tb
import sys
sys.path.append('/home/cuizaixu_lab/fanqingchen/DATA/Code/PLSR_Prediction')
from step1_PLSr.step1_PLSr import Setparameter

def LoadData(datapath, labelpath, dimention, Time,Permutation=0):

    data_list = []
    #Loading Data
    data_files_all = sorted(glob.glob(datapath),reverse=True)
    label_files_all = pd.read_csv(labelpath)
    label = label_files_all[dimention]

    X_train, X_test, y_train, y_test = train_test_split(data_files_all, label, test_size=0.2, random_state=22)

    Train_label = np.array(y_train)
    Test_label = np.array(y_test)
    #Train
    files = X_train[:]
    files_data = []
    for i in files:
        img_data = nib.load(i)
        img_data = img_data.get_data()
        img_data_reshape = tb.upper_tri_indexing(img_data)
        files_data.append(img_data_reshape)

    if Permutation:
        random.shuffle(files_data)
        Train_data = np.asarray(files_data)
    else:
        Train_data = np.asarray(files_data)
    #Test Data
    Test_files = X_test[:]
    Test_list = []
    for j in Test_files:
        test_data = nib.load(j)
        test_data = test_data.get_data()
        test_data_reshape = tb.upper_tri_indexing(test_data)
        Test_list.append(test_data_reshape)

    Test_data = np.asarray(Test_list)
    data_list.append(Train_data)
    data_list.append(Test_data)
    data_list.append(Train_label)
    data_list.append(Test_label)

    return data_list
def PLSPrediction_Model(data_list, dimention,weightpath, Time=1):
    #FC_Prediction
    #bagging,基分类器PLS
    bagging = BaggingRegressor(base_estimator=PLSRegression())

    #网格交叉验证
    cv_times = 5
    param_grid = {'n_estimators': [2, 4, 6,  8, 10]}
    predict_model = GridSearchCV(bagging, param_grid, verbose=6, cv=cv_times)
    #predict_model.coef_()
    predict_model.fit(data_list[0], data_list[2])  #Train_data, Train_label

    # weight
    feature_weight = np.zeros([61776, 1])
    for i, j in enumerate(predict_model.best_estimator_.estimators_):
        print('第{}个模型的系数{}'.format(i, j.coef_))
        # test.to_csv('test_'+str(epoch)+'_'+str(i)+'.csv')
        feature_weight = np.add(j.coef_, feature_weight)

    num = len(predict_model.best_estimator_.estimators_)
    feature_weight_mean = feature_weight / num
    feature_weight_file = pd.DataFrame(feature_weight_mean)
    feature_weight_file.to_csv(weightpath+'feature_weight_Time_' + str(Time) + '.csv')

    print("-best_estimator-", predict_model.best_estimator_, "\n",
          "-best_params-",   predict_model.best_params_, "\n",
          "-best_score-",    predict_model.best_score_
              )


    Predict_Score = predict_model.predict(data_list[1])#test_data
    Predict_Score_new = np.transpose(Predict_Score)
    Corr = np.corrcoef(Predict_Score_new, data_list[3])#test_label

    MAE_inv = np.mean(np.abs(Predict_Score - data_list[3]))#test_label
    print('Prediction Result\n', Predict_Score)
    print('Correlation\n', Corr)
    print('MAE:', MAE_inv)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'Method:PLSRegression_bagging\n')

    tb.ToolboxCSV_server('Predict_Score_bagging' + dimention + '_' + str(Time) + '.csv', Predict_Score)

parameter = Setparameter()
data_list = LoadData(
    parameter['datapath'],
    parameter['labelpath'],
    parameter['dimention'],
    parameter['Time'],
    parameter['Permutation']
)
PLSPrediction_Model(data_list, parameter['dimention'],parameter['weightpath'], parameter['Time'])

