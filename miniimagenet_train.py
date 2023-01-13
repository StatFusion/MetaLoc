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

from meta import Meta


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h

'''
def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    config = [
        # First parameter: number of filters in this layer; Second: width; Third and forth: length and height
        # of each filter; Fifth: stride; Sixth: Padding
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
        #('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [72, 10 * 2 * 2])
    ]

    device = torch.device('cuda')
    print(device)
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    mini = MiniImagenet('/home/student2/dongzewu/CSI/MetaLoc/CSIimage_23_30_07/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry, batchsz=10000, resize=args.imgsz)
    mini_test = MiniImagenet('/home/student2/dongzewu/CSI/MetaLoc/CSIimage0609/', mode='train', n_way=72, k_shot=40,
                             k_query=args.k_qry, batchsz=100, resize=args.imgsz)

    for epoch in range(args.epoch//10000):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)
        #2500 batches; 4 tasks in each batch; 5 classes in each task; 5 query and K support in each class
        #x_spt: size 4,5,3,50,50
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db): #step: batch
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            accs = maml(x_spt, y_spt, x_qry, y_qry)
            if step % 30 == 0:
                print('step:', step, '\ttraining acc:', accs)
            if step % 500 == 0:  # evaluation
                db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                accs_all_test = []
                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
                    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(accs)
                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                print('Test acc:', accs)
'''

def main():
    torch.manual_seed(21)
    torch.cuda.manual_seed_all(21)
    np.random.seed(21)
    print(args)

    config = [
        # input: 50*50*3
        ('conv2d', [10, 3, 3, 3, 1, 0]),
        # 48*48*10
        ('relu', [True]),
        ('bn', [10]),
        ('max_pool2d', [2, 2, 0]),
        # 24*24*10
        ('conv2d', [10, 10, 3, 3, 1, 0]),
        # 22*22*10
        ('relu', [True]),
        ('bn', [10]),
        ('max_pool2d', [2, 2, 0]),
        # 11*11*10
        ('conv2d', [10, 10, 3, 3, 1, 0]),
        # 9*9*10
        ('relu', [True]),
        ('bn', [10]),
        ('max_pool2d', [2, 2, 0]),
        # 4*4*10
        ('conv2d', [10, 10, 3, 3, 1, 0]),
        # 2*2*10
        ('relu', [True]),
        ('bn', [10]),
        #('max_pool2d', [2, 2, 0]),
        ('flatten', []),
        ('linear', [90, 10 * 2 * 2]),
    ]

    device = torch.device('cuda:2')
    print(device)
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)
    batch_size=90
    # batchsz here means total episode number

    #Warning!!: LOS和NLOS对应的坐标不同，需要更改meta以及meta2的相应坐标计算方法
    mini = CSIImagenet2('/home/student2/jgao/CSI/MetaLoc/CSIimage0714', mode='train', n_way=args.n_way, k_shot=args.k_spt,k_query=args.k_qry, batchsz=10000, resize=args.imgsz)
    '''
    mini_test = CSIImagenetfinetune('/home/student2/jgao/CSI/MetaLoc/CSIimage0831', mode='test', n_way=(args.n_way+1), k_shot=args.k_spt,
                             k_query=args.k_qry, batchsz=90, resize=args.imgsz, csi = pd.read_csv('/home/student2/jgao/CSI/MetaLoc/CSIimage0721/csi.csv',header=None), location = pd.read_csv('/home/student2/jgao/CSI/MetaLoc/RSS/CSV/locationcomp.csv',header=None), csi_test = pd.read_csv('/home/student2/jgao/CSI/MetaLoc/CSIimage0721/csi_test.csv',header=None), date='0721')'''
    
    mini_test = CSIImagenetfinetune2('/home/student2/jgao/CSI/MetaLoc/CSIimage0609', mode='test', n_way=(args.n_way+1), k_shot=args.k_spt,
                             k_query=args.k_qry, batchsz=90, resize=args.imgsz, csi = '/home/student2/jgao/CSI/MetaLoc/CSIimage0609/csi.csv', location = pd.read_csv('/home/student2/jgao/CSI/MetaLoc/RSS/CSV/locationcomp.csv',header=None), csi_test = '/home/student2/jgao/CSI/MetaLoc/CSIimage0609/csi_test.csv')

    result=pd.DataFrame(np.full([1300,13+batch_size],np.nan), columns=['test_error_0','test_error_1','test_error_2','test_error_3','test_error_4','test_error_5',\
    'test_error_6','test_error_7','test_error_8','test_error_9','test_error_10','loc error','std']+['error_data' for i in range(batch_size)])
    
    number=0
    for epoch in range(args.epoch//10000):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=False, num_workers=1, pin_memory=True)
        #x_spt: size 4,5,3,50,50
        Mean=[]
        Std=[]
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            accs = maml(x_spt, y_spt, x_qry, y_qry)
            if step % 30 == 0:
                print('step:', step, '\ttraining acc:', accs)
            if step % 50 == 0:  # evaluation
                db_test = DataLoader(mini_test, 1, shuffle=False, num_workers=1, pin_memory=True)
                accs_all_test = []
                loc = []                
                for x_spt, y_spt, x_qry, y_qry in db_test:
                    #print('555555555555555555555')
                    #print(x_spt.shape)
                    
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
                    #print(x_spt.shape)
                    #print(x_spt[:4])
                    #print(x_spt[:-1]) 
                    accs,accs_test_mean = maml.finetunning(x_spt[:args.n_way*args.k_spt], y_spt[:args.n_way*args.k_spt], x_qry[:args.n_way*args.k_qry], y_qry[:args.n_way*args.k_qry],x_spt[args.n_way*args.k_spt:], y_spt[args.n_way*args.k_spt:])
                    accs_all_test.append(accs)  #accs_all_test包含所有accs,这里的accs是每个task的十步的test error
                    loc.append(accs_test_mean)  #loc包含每个task的localization error; acc_test_mean是每个task里的localization error的均值
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

    result.to_csv('/home/student2/dongzewu/CSI/MAMLTS-LOS/0609_new.csv')
            

                
   


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
    args = argparser.parse_args()
    
    main()
