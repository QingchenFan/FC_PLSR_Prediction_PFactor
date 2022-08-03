import random

from datetime import datetime
import os

from step1_PLSr import PLSc_RandomCV_MultiTimes, Setparameter

codepath = '/home/cuizaixu_lab/fanqingchen/DATA/Code/PLSR_Prediction/FC_Prediction'



# serverset=[1,2,3,4,5,6]
# savepath = './'
# labelpath = './'
# dimetion = 'abcdef'
# CVRepeatTimes = 3
# Permutation = 1
# scriptpath = '/Users/fan/PycharmProjects/fc_prediction_rep/abc'
parameter = Setparameter()
data_list = LoadData(
    parameter['datapath'],
    parameter['labelpath'],
    parameter['dimention'],
    parameter['Time'],
    parameter['Permutation']
)
PLSPrediction_Model(data_list, parameter["Permutation"], parameter['Time'])