import sys
import os
from datetime import datetime
import ToolBox as tb
sys.path.append('/home/cuizaixu_lab/fanqingchen/DATA/Code/PLSR_Prediction')
def Setparameter():
    '''
      parameter setting
    :return:dict

    '''
    serverset = ['fanqingchen', 1, 1, 30, 6000, 'lab_fat_c']
    sersavepath = '/home/cuizaixu_lab/fanqingchen/DATA/Res/server_note/'
    scriptpath = '/home/cuizaixu_lab/fanqingchen/DATA/Code/script'

    #datapath = '/home/cuizaixu_lab/fanqingchen/DATA/data/ABCD_FC/ABCD_FC_10min/*.nii' #.nii data
    datapath = '/GPFS/cuizaixu_lab_permanent/fanqingchen/Code/FC_PLSR_Prediction/Model/featurn.txt'  # feture matrix
    labelpath = '/home/cuizaixu_lab/fanqingchen/DATA/data/ABCD_FC/Label/ABCD_Label.csv'
    weightpath = '/home/cuizaixu_lab/fanqingchen/DATA/Res/model_weight/'

    dimention = 'General'  #General Ext ADHD Int Age
    Permutation = 0  # 1: Permutation test   0: no
    kfold = 5  # 1:KFold 0:no
    CVRepeatTimes = 21
    Time = 38  # 0 : test

    setparameter = {
            'serverset':      serverset,
          'sersavepath':    sersavepath,
           'scriptpath':     scriptpath,
        'CVRepeatTimes':  CVRepeatTimes,
             'datapath':       datapath,
            'labelpath':      labelpath,
           'weightpath':     weightpath,
            'dimention':      dimention,
          'Permutation':    Permutation,
                 'Time':           Time,
                'KFold':          kfold
    }
    return setparameter

def PLSc_RandomCV_MultiTimes(serverset, sersavepath, scriptpath, CVRepeatTimes, kfold, dimention,Time,Permutation=0):
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
    else:
        system_cmd = 'python /home/cuizaixu_lab/fanqingchen/DATA/Code/PLSR_Prediction/FC_Prediction/fc_plsr_prediction_bagging_p.py'
    if Permutation == 0:
        scriptfold = scriptpath + '/' + str(datetime.now().strftime('%Y_%m_%d'))+'_'+dimention
        if os.path.exists(scriptfold):
            return
        if not os.path.exists(scriptfold):
            os.makedirs(scriptfold)

        servernotepath = sersavepath + str(datetime.now().strftime('%Y_%m_%d'))+'_'+str(Time)+'_'+dimention
        if not os.path.exists(servernotepath):
            os.makedirs(servernotepath)

        for i in range(CVRepeatTimes):
            count = tb.countnum_2()
            script = open(scriptfold + '/' + 'Time_' + str(i) + '_' + str(count) + '_script.sh', mode='w')
            script.write(Sbatch_Para)
            script.write('\n')
            script.write('#SBATCH -o ' + servernotepath + '/'+'Time_' + str(i) + '_' + str(count) + '_job.%j.out\n')
            script.write('#SBATCH -e ' + servernotepath + '/'+'Time_' + str(i) + '_' + str(count) + '_job.%j.error.txt\n\n')
            script.write(system_cmd)
            script.close()
            os.system('chmod +x ' + scriptfold + '/' + 'Time_' + str(i) + '_' + str(count) + '_script.sh')
            os.system('sbatch ' + scriptfold + '/' + 'Time_' + str(i) + '_' + str(count) + '_script.sh')
    else:
        count = tb.countnum_2()
        scriptfold = scriptpath + '/' + str(datetime.now().strftime('%Y_%m_%d'))
        if os.path.exists(scriptfold):
            return
        if not os.path.exists(scriptfold):
            os.makedirs(scriptfold)

        servernotepath = sersavepath + str(datetime.now().strftime('%Y_%m_%d')) + '_' + str(Time) + '_' + dimention
        if not os.path.exists(servernotepath):
            os.makedirs(servernotepath)
        for i in range(CVRepeatTimes):
            script = open(scriptfold + '/' + 'Time_' + str(i) + '_' + str(count) + '_script.sh', mode='w')
            script.write(Sbatch_Para)
            script.write('\n')
            script.write('#SBATCH -o ' + servernotepath + 'Time_' + str(i) + '_' + str(count) + '_job.%j.out\n')
            script.write('#SBATCH -e ' + servernotepath + 'Time_' + str(i) + '_' + str(count) + '_job.%j.error.txt\n\n')
            script.write(system_cmd)
            script.close()
            os.system('chmod +x ' + scriptfold + '/' + 'Time_' + str(i) + '_' + str(count) + '_script.sh')
            os.system('sbatch ' + scriptfold + '/' + 'Time_' + str(i) + '_' + str(count) + '_script.sh')

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

