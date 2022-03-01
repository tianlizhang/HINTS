import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


def show_err(pred_path, label_path):
    pred = np.loadtxt(pred_path)
    gt = np.loadtxt(label_path)

    o_gt = np.exp(gt)-1
    gt[:,0:1] = o_gt[:,0:1]
    gt[:,1:2] = o_gt[:,1:2]-o_gt[:,0:1]
    gt[:,2:3] = o_gt[:,2:3]-o_gt[:,1:2]
    gt[:,3:4] = o_gt[:,3:4]-o_gt[:,2:3]
    gt[:,4:5] = o_gt[:,4:5]-o_gt[:,3:4]
    gt = np.log(gt+1)

    o_pred = np.exp(pred)-1
    pred[:,0:1] = o_pred[:,0:1]
    pred[:,1:2] = o_pred[:,1:2]-o_pred[:,0:1]
    pred[:,2:3] = o_pred[:,2:3]-o_pred[:,1:2]
    pred[:,3:4] = o_pred[:,3:4]-o_pred[:,2:3]
    pred[:,4:5] = o_pred[:,4:5]-o_pred[:,3:4]
    pred = np.log(pred+1)
    pred = np.nan_to_num(pred)

    num = min(len(pred),len(gt))

    # year1 = pred[:num,:1],gt[:num,:1]
    # year2 = pred[:num,1:2],gt[:num,1:2]
    # year3 = pred[:num,2:3],gt[:num,2:3]
    # year4 = pred[:num,3:4],gt[:num,3:4]
    # year5 = pred[:num,4:],gt[:num,4:]
    pred = pred[:num,:]
    gt = gt[:num,:]

    # print ("1st Year MALE:{}  RMSLE:{}".format(mean_absolute_error(year1[0],year1[1]),np.sqrt(mean_squared_error(year1[0],year1[1]))))
    # print ("2nd Year MALE:{}  RMSLE:{}".format(mean_absolute_error(year2[0],year2[1]),np.sqrt(mean_squared_error(year2[0],year2[1]))))
    # print ("3th Year MALE:{}  RMSLE:{}".format(mean_absolute_error(year3[0],year3[1]),np.sqrt(mean_squared_error(year3[0],year3[1]))))
    # print ("4th Year MALE:{}  RMSLE:{}".format(mean_absolute_error(year4[0],year4[1]),np.sqrt(mean_squared_error(year4[0],year4[1]))))
    # print ("5th Year MALE:{}  RMSLE:{}".format(mean_absolute_error(year5[0],year5[1]),np.sqrt(mean_squared_error(year5[0],year5[1]))))
    print ("Overall MALE:{}  RMSLE:{}".format(mean_absolute_error(pred,gt),np.sqrt(mean_squared_error(pred,gt))))
    #mean_absolute_error(pred,gt)
    return mean_squared_error(pred,gt)


if __name__ == '__main__':
    datasets = ['aminer', 'aps']
    beta = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    num = [0]

    y = [ []for _ in range(len(datasets)) ]
    for idx, dataset in enumerate(datasets):
        for i in range(len(beta)):
            b = beta[i]
            for n in num:
                pred_path = "./result_beta/"+dataset+"_test_test_beta_"+str(b)+"num_"+str(n)+".txt"
                label_path ="./result_beta/"+dataset+"_test_labels_"+str(b)+"num_"+str(n)+".txt"
                print(f'num = {n}---------------------')
                err = show_err(pred_path, label_path)
                y[idx].append(err)
    
    plt.plot(beta, y[0])
    plt.plot(beta, y[1])
    plt.legend(datasets)
    plt.xlabel('beta')
    plt.ylabel('RMSLE')
    plt.savefig('jpg/01_beta.jpg')

