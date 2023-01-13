import  torch, os
import  numpy as np
import pandas as pd
from    MiniImagenet import CSIImagenet,CSIImagenetfinetune
from    MiniImagenet2 import CSIImagenet2,CSIImagenetfinetune2
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse

from meta2 import Meta


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():
    torch.manual_seed(21)
    torch.cuda.manual_seed_all(21)
    np.random.seed(21)
    print(args)
    
    #CNN structure:
    config = [
        ('conv2d', [10, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [10]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [10, 10, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [10]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [10, 10, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [10]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [10, 10, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [10]),
        #('max_pool2d', [2, 2, 0]),
        ('flatten', []),
        ('linear', [90, 10 * 2 * 2]),
    ]

    device = torch.device('cuda:1')
    print(device)
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)
    batch_size=90
    # batchsz here means total episode number

    mini=[]     #mini stores all the training domains' data
    for i in args.date:
        mini_date=CSIImagenet2('/home/student2/jgao/CSI/MetaLoc/CSIimage'+i, mode='train', n_way=args.n_way, k_shot=args.k_spt,k_query=args.k_qry, batchsz=10000, resize=args.imgsz)
        mini.append(mini_date)
    #mini_test stores the testing data
    mini_test = CSIImagenetfinetune2('/home/student2/jgao/CSI/MetaLoc/CSIimage_21_19_31_24_25', mode='test', n_way=(args.n_way+1), k_shot=args.k_spt,
                             k_query=args.k_qry, batchsz=90, resize=args.imgsz, csi = '/home/student2/jgao/CSI/MetaLoc/CSIimage_21_19_31_24_25/csi.csv', location = pd.read_csv('/home/student2/jgao/CSI/MetaLoc/RSS/CSV/locationcomp.csv',header=None), csi_test = '/home/student2/jgao/CSI/MetaLoc/CSIimage_21_19_31_24_25/csi_test.csv')

    result=pd.DataFrame(np.full([1300,13+batch_size],np.nan), columns=['test_error_0','test_error_1','test_error_2','test_error_3','test_error_4','test_error_5',\
    'test_error_6','test_error_7','test_error_8','test_error_9','test_error_10','loc error','std']+['error_data' for i in range(batch_size)])
    #test_error_i represents the averaged test error on the query set of the training task after the i-th gradient descent step
    #loc error represents the localization error tested on the testing task

    number=0
    for epoch in range(args.epoch//10000):
        # fetch meta_batchsz num of episode each time
        for step in range(2501):
            lst=[0,1,2,3,4]   #The length of the lst should be the number of training domains
            a=random.choice(lst)
            lst.remove(a)
            b=random.choice(lst)
            #Randomly pick out two training domains in each iteration step
            db = DataLoader(mini[a], args.task_num, shuffle=False, num_workers=1, pin_memory=True)
            db2 = DataLoader(mini[b], args.task_num, shuffle=True, num_workers=1, pin_memory=True)
            #x_spt: size 4,5,3,50,50
            Mean=[]
            Std=[]
            for step1, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
                for step2, (x_spt2, y_spt2, x_qry2, y_qry2) in enumerate(db2):
                    if step2==0:
                        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
                        x_spt2, y_spt2, x_qry2, y_qry2 = x_spt2.to(device), y_spt2.to(device), x_qry2.to(device), y_qry2.to(device)
                        accs = maml(x_spt, y_spt, x_qry, y_qry,x_spt2, y_spt2, x_qry2, y_qry2)
                    else:
                        break
                break
            if step % 30 == 0:
                print('step:', step, '\ttraining acc:', accs)
            if step % 50 == 0:  # evaluation
                db_test = DataLoader(mini_test, 1, shuffle=False, num_workers=1, pin_memory=True)
                accs_all_test = []
                loc = []                
                for x_spt, y_spt, x_qry, y_qry in db_test:                    
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
                    accs,accs_test_mean = maml.finetunning(x_spt[:args.n_way*args.k_spt], y_spt[:args.n_way*args.k_spt], x_qry[:args.n_way*args.k_qry], y_qry[:args.n_way*args.k_qry],x_spt[args.n_way*args.k_spt:], y_spt[args.n_way*args.k_spt:])
                    accs_all_test.append(accs)  
                    loc.append(accs_test_mean)  
                    # accs: the test errors of each training task with respect to the number of gradient steps
                    # accs_test_mean: the averaged localization error
                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16) #这个accs指此轮测试(每50步数)中平均下来的十步的test error
                accs_test_mean = np.array(loc).mean(axis=0).astype(np.float16) #报告的loc error即为loc的均值
                Mean.append(accs_test_mean)
                std = np.array(loc).std(axis=0).astype(np.float16)
                Std.append(std)
                print('Test error:', accs)
                print('Localization error:', accs_test_mean)
                print('Localization std:', std)
                result.loc[number,'test_error_0':'test_error_10']=list(accs)
                result.loc[number,'loc error']=accs_test_mean
                result.loc[number,'std']=std
                result.loc[number,'error_data':]=loc
                number+=1
        print('Localization error list:', Mean)
        print('Localization std list:', Std)
    #The following code is used to plot the probability density
    '''
    data=[]
    for i in range(len(list7)):
        data.append(np.sqrt(np.sum((list7[i] - label[i,:])**2)))
    data = np.array(data).flatten()
    print(len(data))
    print(data.std())
    data_copy=pd.DataFrame(data)
    data_copy.to_csv('/home/student2/dongzewu/CSI/ILCL/Incremental-Learning-for-indoor-localization/cdf/3_0714.csv')
    ecdf = sm.distributions.ECDF(data)
    #等差数列，用于绘制X轴数据
    x = np.linspace(min(data), max(data))
    # x轴数据上值对应的累计密度概率
    y = ecdf(x)
    plt.step(x, y)
    plt.show()
    plt.savefig("/home/student2/dongzewu/CSI/ILCL/Incremental-Learning-for-indoor-localization/cdf/3_cdf_0714.jpg")'''

    result.to_csv('/home/student2/dongzewu/CSI/MAMLDG-LOS to NLOS/result_RGB.csv')


                
   


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    #argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=10)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=3)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=50)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--date', type=int, help='update steps for finetunning', default=['0609','0623','0630','0707','0714'])
    argparser.add_argument('--weight', type=int, help='update steps for finetunning', default=1)
    #'0925','0721','0831','0819'
    #'0609','0623','0630','0707','0714'
    args = argparser.parse_args()
    
    main()
