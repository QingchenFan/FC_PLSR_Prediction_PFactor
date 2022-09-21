import numpy as np
from sklearn.decomposition import PCA
import nibabel as nib
import glob
import csv as csv
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import GridSearchCV
import joblib
import ToolBox as tb
import xgboost as xgb
Times = 22
dimention = 'General'    #['General', 'Int', 'ADHD', 'Ext'] #[]
label_files_all = pd.read_csv("/home/cuizaixu_lab/fanqingchen/DATA/data/ABCD_FC/Label/ABCD_Label.csv")
label = label_files_all[dimention]
#Label
y_label = np.array(label)

x_data = np.loadtxt('/GPFS/cuizaixu_lab_permanent/fanqingchen/Code/FC_PLSR_Prediction/Model/featurn.txt')
outer_results_R = []
outer_results_mae = []
outer_results_r2 = []
epoch = 0
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(x_data):
    epoch = epoch + 1
    # split data
    X_train, X_test = x_data[train_index, :], x_data[test_index, :]
    y_train, y_test = y_label[train_index], y_label[test_index]

    # Model
    Hyper_param = {'max_depth': range(3, 10, 2)}

    predict_model = GridSearchCV(estimator=xgb.XGBRegressor(booster='gbtree',
                                                            learning_rate=0.1,
                                                            n_estimators=160,
                                                            verbosity=0,
                                                            objective='reg:linear'),
                                 param_grid=Hyper_param,
                                 scoring='neg_mean_absolute_error',
                                 verbose=1,
                                 cv=5)
    predict_model.fit(X_train, y_train)

    Predict_Score = predict_model.predict(X_test)

    print("-best_estimator-", predict_model.best_estimator_, "-",
          "-best_params-", predict_model.best_params_, "-",
          "-best_score-", predict_model.best_score_
          )
    # save model param

    Corr = np.corrcoef(Predict_Score.T, y_test)
    Corr = Corr[0, 1]
    outer_results_R.append(Corr)

    MAE_inv = np.mean(np.abs(Predict_Score - y_test))
    print('Prediction Result\n', Predict_Score)
    print('Correlation\n', Corr)
    print('MAE:', MAE_inv)
print('Result: R=%.3f' % (np.mean(outer_results_R)))



