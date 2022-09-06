import sys
import os
import time
from datetime import datetime
import ToolBox as tb
sys.path.append('/home/cuizaixu_lab/fanqingchen/DATA/Code/PLSR_Prediction')
def Setparameter():
    '''
      parameter setting
    :return:dict

    '''
    # the path of saving file
    serverset = ['fanqingchen', 1, 1, 10, 8000, 'lab_fat_c']
    sersavepath = '/home/cuizaixu_lab/fanqingchen/DATA/Res/HCP_Res/server_note/'
    scriptpath = '/home/cuizaixu_lab/fanqingchen/DATA/Code/script/HCP_script'
    weightpath = '/home/cuizaixu_lab/fanqingchen/DATA/Res/HCP_Res/model_weight/'
    # loading data path
    #datapath = '/home/cuizaixu_lab/fanqingchen/DATA/data/ABCD_FC/ABCD_FC_10min/*.nii'   #.nii data
    datapath = '/home/cuizaixu_lab/fanqingchen/DATA/data/Feature_Matrix/HCPfeature_0904.txt'  # feture matrix
    labelpath = '/home/cuizaixu_lab/fanqingchen/DATA/data/ABCD_HCP2016_FC/Label/HCPLable0904.csv'
    covariatespath = '/home/cuizaixu_lab/fanqingchen/DATA/data/ABCD_HCP2016_FC/Label/ageSexFd0904.csv'

    Permutation = 0        # 1: Permutation test   0: no
    # TODO
    dimention = 'General'  # General Ext ADHD Int Age
    dataMark = 'HCP'
    kfold = 2              # 1:KFold 0:no
    CVRepeatTimes = 101
    Time = 71

    setparameter = {
            'serverset':         serverset,
          'sersavepath':       sersavepath,
           'scriptpath':        scriptpath,
        'CVRepeatTimes':     CVRepeatTimes,
             'datapath':          datapath,
            'labelpath':         labelpath,
           'weightpath':        weightpath,
            'dimention':         dimention,
          'Permutation':       Permutation,
       'covariatespath':    covariatespath,
                 'Time':              Time,
                'KFold':             kfold,
             'dataMark':          dataMark
    }
    return setparameter

def PLSc_RandomCV_MultiTimes(serverset, sersavepath, scriptpath, CVRepeatTimes, kfold, dimention, Time, Permutation=0):
    '''
    :param serverset: Server parameter settings
    :param savepath:The result storage path of the server
    :param scriptpath:The path to the script on the server
    :param CVRepeatTimes:Script execution times
    :param Permutation: Whether to replace the test  1：permutation test 0：no permutation test
    :return:
    '''
    Sbatch_Para = '#!/bin/bash\n'+'#SBATCH --qos=lab_fat\n'+'#SBATCH --job-name={}\n#SBATCH --nodes={}\n#SBATCH --ntasks={}\n#SBATCH --cpus-per-task={}\n#SBATCH --mem-per-cpu={}\n#SBATCH -p {}\n'.format(*serverset)
    if kfold:
        system_cmd = 'python /home/cuizaixu_lab/fanqingchen/DATA/Code/PLSR_Prediction/FC_Prediction/fc_kfold_plsr_prediction_bagging.py'
        #system_cmd = 'python /home/cuizaixu_lab/fanqingchen/DATA/Code/PLSR_Prediction/FC_Prediction/fc_kfold_plsr_prediction.py'
        #system_cmd = 'python /home/cuizaixu_lab/fanqingchen/DATA/Code/PLSR_Prediction/FC_Prediction/fc_kfold_svr_prediction.py'

    else:
        system_cmd = 'python /home/cuizaixu_lab/fanqingchen/DATA/Code/PLSR_Prediction/FC_Prediction/fc_plsr_prediction_bagging_p.py'
    if Permutation == 0:
        scriptfold = scriptpath + '/' + str(datetime.now().strftime('%Y_%m_%d'))+'_HCP_'+dimention
        if os.path.exists(scriptfold):
            return
        if not os.path.exists(scriptfold):
            os.makedirs(scriptfold)

        servernotepath = sersavepath + str(datetime.now().strftime('%Y_%m_%d'))+'_HCP_'+str(Time)+'_'+dimention
        if not os.path.exists(servernotepath):
            os.makedirs(servernotepath)

        for i in range(1, CVRepeatTimes+1):

            script = open(scriptfold + '/' + 'Time_' + str(i) + '_' + 'script.sh', mode='w')
            script.write(Sbatch_Para)
            script.write('\n')
            script.write('#SBATCH -o ' + servernotepath + '/'+'Time_' + str(i) + '_' + 'job.%j.out\n')
            script.write('#SBATCH -e ' + servernotepath + '/'+'Time_' + str(i) + '_' + 'job.%j.error.txt\n\n')
            script.write(system_cmd)
            script.close()
            os.system('chmod +x ' + scriptfold + '/' + 'Time_' + str(i) + '_' + 'script.sh')
            time.sleep(1)
            os.system('sbatch ' + scriptfold + '/' + 'Time_' + str(i) + '_' + 'script.sh')
    else:
        scriptfold = scriptpath + '/' + str(datetime.now().strftime('%Y_%m_%d'))+'_'+dimention

        if os.path.exists(scriptfold):
            return
        if not os.path.exists(scriptfold):
            os.makedirs(scriptfold)

        servernotepath = sersavepath + str(datetime.now().strftime('%Y_%m_%d')) + '_HCP_' + str(Time) + '_' + dimention
        if not os.path.exists(servernotepath):
            os.makedirs(servernotepath)
        for i in range(CVRepeatTimes):
            script = open(scriptfold + '/' + 'Time_' + str(i) + '_' + 'script.sh', mode='w')
            script.write(Sbatch_Para)
            script.write('\n')
            script.write('#SBATCH -o ' + servernotepath + 'Time_' + str(i) + '_' + 'job.%j.out\n')
            script.write('#SBATCH -e ' + servernotepath + 'Time_' + str(i) + '_' + 'job.%j.error.txt\n\n')
            script.write(system_cmd)
            script.close()
            os.system('chmod +x ' + scriptfold + '/' + 'Time_' + str(i) + '_' + 'script.sh')
            time.sleep(1)
            os.system('sbatch ' + scriptfold + '/' + 'Time_' + str(i) + '_' + 'script.sh')

if __name__ == '__main__':
    setparameter = Setparameter()
    PLSc_RandomCV_MultiTimes(
                             setparameter['serverset'],
                             setparameter['sersavepath'],
                             setparameter['scriptpath'],
                             setparameter['CVRepeatTimes'],
                             setparameter['KFold'],
                             setparameter['dimention'],
                             setparameter['Time'],
                             setparameter['Permutation']
                             )

