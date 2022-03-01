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

    pred = pred[:num,:]
    gt = gt[:num,:]

    cut1, cut2, cut3, cut4 = int(0.1*num), int(0.45*num), int(0.55*num), int(0.9*num)
    ls = [ [] for _ in range(6)]
    for j in range(5):
        xx = gt[:, j]
        idx1 = np.argsort(xx, axis=0)[0:cut1]
        idx2 = np.argsort(xx, axis=0)[cut2:cut3]
        idx3 = np.argsort(xx, axis=0)[cut4:]
        # part1 = mean_squared_error(pred[idx1, j:j+1],gt[idx1, j:j+1])
        # part2 = mean_squared_error(pred[idx2, j:j+1],gt[idx2, j:j+1])
        # part3 = mean_squared_error(pred[idx3, j:j+1],gt[idx3, j:j+1])
        p1, l1 = np.mean(pred[idx1, j]), np.mean(gt[idx1, j])
        p2, l2 = np.mean(pred[idx2, j]), np.mean(gt[idx2, j])
        p3, l3 = np.mean(pred[idx3, j]), np.mean(gt[idx3, j])
        
        ls[0].append(p1)
        ls[1].append(l1)
        ls[2].append(p2)
        ls[3].append(l2)
        ls[4].append(p3)
        ls[5].append(l3)
        
        
    # print ("Overall MALE:{}  RMSLE:{}".format(mean_absolute_error(pred,gt),np.sqrt(mean_squared_error(pred,gt))))
    #mean_absolute_error(pred,gt)
    return ls


if __name__ == '__main__':
    datasets = ['aminer']
    beta = [0.5]
    num = [0]

    y = [ []for _ in range(3) ]
    for idx, dataset in enumerate(datasets):
        for i in range(len(beta)):
            b = beta[i]
            for n in num:
                pred_path = "./result_beta/"+dataset+"_test_test_beta_"+str(b)+"num_"+str(n)+".txt"
                label_path ="./result_beta/"+dataset+"_test_labels_"+str(b)+"num_"+str(n)+".txt"
                print(f'num = {n}---------------------')
                lst = show_err(pred_path, label_path)
                
    # for i in range(6):
    # plt.subplot(1, 3)
    xx = [1,2,3,4,5]
    # plt.plot(xx, lst[0])
    # plt.plot(xx, lst[1])
    plt.plot(xx, np.abs(np.array(lst[1])-np.array(lst[0])))
    # plt.legend(['0th-10th, pred', '0th-10th, true'])s
    # plt.savefig('jpg/02_rank_01.jpg')

    # plt.figure()
    # plt.plot(xx, lst[2])
    # plt.plot(xx, lst[3])
    plt.plot(xx, np.abs(np.array(lst[3])-np.array(lst[2])))
    # plt.legend(['45th-55th, pred', '45th-55th, true'])
    # plt.savefig('jpg/02_rank_02.jpg')

    # plt.figure()
    # plt.plot(xx, lst[4])
    # plt.plot(xx, lst[5])
    plt.plot(xx, np.abs(np.array(lst[5])-np.array(lst[4])))
    # plt.legend(['0th-10th, pred', '0th-10th, true', \
    #     '45th-55th, pred', '45th-55th, true', '90th-99th, pred', '90th-99th, true'])
    plt.title('Beta')
    plt.xlabel('Year')
    plt.ylabel('Loss')
    plt.legend(['0th-10th, loss', \
        '45th-55th, loss', '90th-99th, loss'])
    plt.savefig('jpg/02_rank_beta_loss.jpg')
    # plt.plot(lst[2], lst[3])
    # plt.plot(lst[4], lst[5])
    # plt.plot(beta, y[1])
    # plt.legend(datasets)
    # plt.xlabel('beta')
    # plt.ylabel('RMSLE')
    # plt.savefig('jpg/02_rank.jpg')

