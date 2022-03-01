import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
# import tensorflow as tf
import math


def show_err(pred_path, label_path):
    pred = np.loadtxt(pred_path)
    gt = np.loadtxt(label_path)
    

    o_gt = np.exp(gt)-1
    gt[:,0:1] = o_gt[:,0:1]
    gt[:,1:2] = o_gt[:,1:2]-o_gt[:,0:1]
    gt[:,2:3] = o_gt[:,2:3]-o_gt[:,1:2]
    gt[:,3:4] = o_gt[:,3:4]-o_gt[:,2:3]
    gt[:,4:5] = o_gt[:,4:5]-o_gt[:,3:4]
    gt = np.log(gt+1.0)/np.log(2.0) ### log以2为底

    o_pred = np.exp(pred)-1
    pred[:,0:1] = o_pred[:,0:1]
    pred[:,1:2] = o_pred[:,1:2]-o_pred[:,0:1]
    pred[:,2:3] = o_pred[:,2:3]-o_pred[:,1:2]
    pred[:,3:4] = o_pred[:,3:4]-o_pred[:,2:3]
    pred[:,4:5] = o_pred[:,4:5]-o_pred[:,3:4]
    pred = np.log(pred+1.0)/np.log(2.0) ### log以2为底
    pred, gt = np.nan_to_num(pred), np.nan_to_num(gt)

    num = min(len(pred),len(gt))

    year1 = pred[:num,:1],gt[:num,:1]
    year2 = pred[:num,1:2],gt[:num,1:2]
    year3 = pred[:num,2:3],gt[:num,2:3]
    year4 = pred[:num,3:4],gt[:num,3:4]
    year5 = pred[:num,4:],gt[:num,4:]
    pred = pred[:num,:]
    gt = gt[:num,:]

    print ("1st Year MALE:{}  RMSLE:{}".format(mean_absolute_error(year1[0],year1[1]),np.sqrt(mean_squared_error(year1[0],year1[1]))))
    print ("2nd Year MALE:{}  RMSLE:{}".format(mean_absolute_error(year2[0],year2[1]),np.sqrt(mean_squared_error(year2[0],year2[1]))))
    print ("3th Year MALE:{}  RMSLE:{}".format(mean_absolute_error(year3[0],year3[1]),np.sqrt(mean_squared_error(year3[0],year3[1]))))
    print ("4th Year MALE:{}  RMSLE:{}".format(mean_absolute_error(year4[0],year4[1]),np.sqrt(mean_squared_error(year4[0],year4[1]))))
    print ("5th Year MALE:{}  RMSLE:{}".format(mean_absolute_error(year5[0],year5[1]),np.sqrt(mean_squared_error(year5[0],year5[1]))))
    print ("Overall MALE:{}  RMSLE:{}".format(mean_absolute_error(pred,gt),np.sqrt(mean_squared_error(pred,gt))))

    # mm = np.zeros(12)
    # mm[0], mm[1] = mean_absolute_error(pred,gt),np.sqrt(mean_squared_error(pred,gt))
    # mm[2], mm[3] = mean_absolute_error(year1[0],year1[1]),np.sqrt(mean_squared_error(year1[0],year1[1]))
    # mm[4], mm[5] = mean_absolute_error(year2[0],year2[1]),np.sqrt(mean_squared_error(year2[0],year2[1]))
    # mm[6], mm[7] = mean_absolute_error(year3[0],year3[1]),np.sqrt(mean_squared_error(year3[0],year3[1]))
    # mm[8], mm[9] = mean_absolute_error(year4[0],year4[1]),np.sqrt(mean_squared_error(year4[0],year4[1]))
    # mm[10], mm[11] = mean_absolute_error(year5[0],year5[1]), np.sqrt(mean_squared_error(year5[0],year5[1]))
    # mm = tf.reduce_mean(tf.pow(pred - gt, 2))
    mm = np.mean( (pred - gt)*(pred-gt) )
    # print(f'mm: {mm}')
    # print(ztl)
    return mm

if __name__ == '__main__':
    # dataset = 'v13' #'aps'
    dataset = 'aminer'
    beta = [0.5]
    # num = [0,1,2]
    num = [0,1,2,4,5,6,9]

    err = np.zeros(12)
    cnt = 0
    for i in range(len(beta)):
        b = beta[i]
        for n in num:
            pred_path = "../result/"+dataset+"_test_test_beta_"+str(b)+"num_"+str(n)+".txt"
            label_path ="../result/"+dataset+"_test_labels_"+str(b)+"num_"+str(n)+".txt"
            print(f'num = {n}---------------------')
            err += show_err(pred_path, label_path)
            cnt += 1
            
    
    mean_err = err/cnt
    print('mean_err', mean_err)