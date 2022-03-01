import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys

def show_err(pred_path, label_path):
    pred = np.loadtxt(pred_path)
    gt = np.loadtxt(label_path)
    

    o_gt = np.exp(gt)-1 # exp，再累减 == label先累加再log
    gt[:,0:1] = o_gt[:,0:1]
    gt[:,1:2] = o_gt[:,1:2]-o_gt[:,0:1]
    gt[:,2:3] = o_gt[:,2:3]-o_gt[:,1:2]
    gt[:,3:4] = o_gt[:,3:4]-o_gt[:,2:3]
    gt[:,4:5] = o_gt[:,4:5]-o_gt[:,3:4]
    gt = np.log(gt+1) # log to calc MALE(Mean Absolute Log-scaled Error)

    o_pred = np.exp(pred)-1
    pred[:,0:1] = o_pred[:,0:1]
    pred[:,1:2] = o_pred[:,1:2]-o_pred[:,0:1]
    pred[:,2:3] = o_pred[:,2:3]-o_pred[:,1:2]
    pred[:,3:4] = o_pred[:,3:4]-o_pred[:,2:3]
    pred[:,4:5] = o_pred[:,4:5]-o_pred[:,3:4]

    pred[pred<=0] = 0
    pred = np.log(pred+1)
    pred, gt = np.nan_to_num(pred), np.nan_to_num(gt)

    num = min(len(pred),len(gt))

    year1 = pred[:num,:1],gt[:num,:1]
    year2 = pred[:num,1:2],gt[:num,1:2]
    year3 = pred[:num,2:3],gt[:num,2:3]
    year4 = pred[:num,3:4],gt[:num,3:4]
    year5 = pred[:num,4:],gt[:num,4:]
    pred = pred[:num,:]
    gt = gt[:num,:]

    # print ("1st Year MALE:{}  RMSLE:{}".format(mean_absolute_error(year1[0],year1[1]),np.sqrt(mean_squared_error(year1[0],year1[1]))))
    # print ("2nd Year MALE:{}  RMSLE:{}".format(mean_absolute_error(year2[0],year2[1]),np.sqrt(mean_squared_error(year2[0],year2[1]))))
    # print ("3th Year MALE:{}  RMSLE:{}".format(mean_absolute_error(year3[0],year3[1]),np.sqrt(mean_squared_error(year3[0],year3[1]))))
    # print ("4th Year MALE:{}  RMSLE:{}".format(mean_absolute_error(year4[0],year4[1]),np.sqrt(mean_squared_error(year4[0],year4[1]))))
    # print ("5th Year MALE:{}  RMSLE:{}".format(mean_absolute_error(year5[0],year5[1]),np.sqrt(mean_squared_error(year5[0],year5[1]))))
    # print ("Overall MALE:{}  RMSLE:{}".format(mean_absolute_error(pred,gt),np.sqrt(mean_squared_error(pred,gt))))

    male, rmsle = np.zeros(6), np.zeros(6)
    male[0], rmsle[0] = mean_absolute_error(pred,gt),np.sqrt(mean_squared_error(pred,gt))
    male[1], rmsle[1] = mean_absolute_error(year1[0],year1[1]),np.sqrt(mean_squared_error(year1[0],year1[1]))
    male[2], rmsle[2] = mean_absolute_error(year2[0],year2[1]),np.sqrt(mean_squared_error(year2[0],year2[1]))
    male[3], rmsle[3] = mean_absolute_error(year3[0],year3[1]),np.sqrt(mean_squared_error(year3[0],year3[1]))
    male[4], rmsle[4] = mean_absolute_error(year4[0],year4[1]),np.sqrt(mean_squared_error(year4[0],year4[1]))
    male[5], rmsle[5] = mean_absolute_error(year5[0],year5[1]), np.sqrt(mean_squared_error(year5[0],year5[1]))
    return male, rmsle

if __name__ == '__main__':
    try:
        model = str(sys.argv[1]) # ['gcn', 'seq2seq', 'HINTS']
        dataset = str(sys.argv[2]) # ['aminer', 'aps']
    except:
        model = 'HINTS'
        dataset = 'aminer'
    print('model:', model, ' dataset:', dataset)
    
    if model == 'gcn':
        SPATH = '../experiment/result_gcn/'
    elif model == 'seq2seq':
        SPATH = '../experiment/result_seq2seq/'
    else:
        SPATH = '../result/'

    beta = [0.5]
    num = [0,1,2,3,4,5,6,7,8,9]

    MALE, RMSLE = np.zeros(6), np.zeros(6)
    cnt = 0
    for i in range(len(beta)):
        b = beta[i]
        for n in num:
            pred_path = SPATH + dataset+"_test_test_beta_"+str(b)+"num_"+str(n)+".txt"
            label_path = SPATH + dataset+"_test_labels_"+str(b)+"num_"+str(n)+".txt"

            male, rmsle = show_err(pred_path, label_path)
            MALE = MALE + male
            RMSLE = RMSLE + rmsle
            cnt += 1
    
    MALE = MALE/cnt
    RMSLE = RMSLE/cnt

    print ("Overall MALE:{}  RMSLE:{}".format(MALE[0], RMSLE[0]))
    print ("1st Year MALE:{}  RMSLE:{}".format(MALE[1], RMSLE[1]))
    print ("2st Year MALE:{}  RMSLE:{}".format(MALE[2], RMSLE[2]))
    print ("3st Year MALE:{}  RMSLE:{}".format(MALE[3], RMSLE[3]))
    print ("4st Year MALE:{}  RMSLE:{}".format(MALE[4], RMSLE[4]))
    print ("5st Year MALE:{}  RMSLE:{}".format(MALE[5], RMSLE[5]))