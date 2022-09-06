# import random
# import math
# from sklearn.cross_decomposition import CCA
# import numpy as np
# X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]]
# y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
# y = np.array(y)
# print(y)
# np.random.shuffle(y)
# print(y)
# list_1 = []
#
# list_1.append(y)
# a = list_1[0]
# print('--a--',a)
# # cca = CCA(n_components=1)
# # cca.fit(X, Y)
# # CCA(n_components=1)
# # X_c, Y_c = cca.transform(X, Y)
# # res = np.dot(Y, cca._coef_)
# # print(cca.coef_)
# # print(Y,type(Y))
# # print(cca.y_loadings_)
# # print('Y_c:',Y_c)
# #
# # print('res:',res)
# print(2**3)
# params = dict(C=[math.pow(2, i) for i in range(-10, 10)])
# print(params)
import glob
datapath = '/home/cuizaixu_lab/fanqingchen/DATA/data/ABCD_HCP2016/*.nii'
data_files_all = sorted(glob.glob(datapath), reverse=True)
list_all = []
for i in data_files_all:
    list_all.append(i)
fw = open("/home/cuizaixu_lab/fanqingchen/DATA/data/ABCD_HCP_Label.csv", mode='w')
for j in list_all:

    fw.write(j)
    fw.write('\n')
