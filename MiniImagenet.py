from cProfile import label
import os
from signal import raise_signal
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random
import csv
import cv2
from PIL import Image
import requests
from io import BytesIO
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class CSIImagenet(Dataset):
    """
    put mini-imagenet files as :
    root :
        |- images/*.jpg includes all imgeas
        |- train.csv
        |- test.csv
        |- val.csv
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize):
        """

        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        """

        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize to
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (
        mode, batchsz, n_way, k_shot, k_query, resize))

        if mode == 'train':
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 # transforms.RandomHorizontalFlip(),
                                                 # transforms.RandomRotation(5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
        else:
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

        self.path = os.path.join(root, 'sample10', 'RGB')  # image path
        csvdata = self.loadCSV(os.path.join(root, 'sample10', 'label' + '.csv'))  # csv path
        self.data = []
        self.img2label = {}
        for i, (k, v) in enumerate(csvdata.items()):
            self.data.append(v)  # [[img1, img2, ...], 
            self.img2label[k] = k  # {"img_name[:9]":label}

        self.cls_num = len(self.data)#the number of class in train/
        self.create_batch(self.batchsz)

    def loadCSV(self, csvf):
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        """
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            #next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                # append filename to current label
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels

    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly from all classes
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets
    
    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        # [setsz, 3, resize, resize]
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        # [setsz]
        support_y = np.zeros((self.setsz), dtype=np.int)
        # [querysz, 3, resize, resize]
        query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
        # [querysz]
        query_y = np.zeros((self.querysz), dtype=np.int)

        flatten_support_x = [os.path.join(self.path, item)
                             for sublist in self.support_x_batch[index] for item in sublist]
        

        support_y = np.array(
            [item.split('_')[0]  # filename:n0153282900000005.jpg, the first 9 characters treated as label
             for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)

        flatten_query_x = [os.path.join(self.path, item)
                           for sublist in self.query_x_batch[index] for item in sublist]
        query_y = np.array([item.split('_')[0]
                            for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)

 
        unique = np.unique(support_y)
        random.shuffle(unique)
        # relative means the label ranges from 0 to n-way
        '''
        support_y_relative = np.zeros(self.setsz)
        query_y_relative = np.zeros(self.querysz)
        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx
            query_y_relative[query_y == l] = idx'''

        for i, path in enumerate(flatten_support_x):
            support_x[i] = self.transform(path)
            
        for i, path in enumerate(flatten_query_x):
            query_x[i] = self.transform(path)

        return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz

class CSIImagenetfinetune(Dataset):
    """
    put mini-imagenet files as :
    root :
        |- images/*.jpg includes all imgeas
        |- train.csv
        |- test.csv
        |- val.csv
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize, csi_test, csi,location,date):
        """

        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        """
        self.csi_test = csi_test
        self.csi = csi 
        self.location = location 
        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize to
        self.date=date
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (
        mode, batchsz, n_way, k_shot, k_query, resize))

        if mode == 'train':
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 # transforms.RandomHorizontalFlip(),
                                                 # transforms.RandomRotation(5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
        else:
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

        self.path = os.path.join(root, 'sample10', 'RGB')  # image path
        csvdata = self.loadCSV(os.path.join(root, 'sample10', 'label' + '.csv'))  # csv path
        self.data = []
        self.img2label = {}
        for i, (k, v) in enumerate(csvdata.items()):
            #print (k)
            #print("11111111111111")
#             print ("11111111111")
#             print(v)
            self.data.append(v)  # [[img1, img2, ...], [img111, ...]]
#             print (i)
#             print("11111111111111")
#             print(self.startidx)
            self.img2label[k] = k  # {"img_name[:9]":label}
            
            # print(k) #label
            # print(v) #['0_1.png', '0_2.png', '0_3.png', '0_4.png', '0_5.png', '0_6.png', '0_7.png', '0_8.png', '0_9.png']
        self.cls_num = len(self.data)#the number of class in train/test dataset
        #print(self.cls_num)
        self.create_batch(self.batchsz)

    def loadCSV(self, csvf):
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        """
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            #next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                # append filename to current label
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels
    
    # def create_batch(self, batchsz):
    #     """
    #     create batch for meta-learning.
    #     ×episode× here means batch, and it means how many sets we want to retain.
    #     :param episodes: batch size
    #     :return:
    #     """
    #     self.support_x_batch = []  # support set batch
    #     self.query_x_batch = []  # query set batch
    #     for b in range(batchsz):  # for each batch
    #         # 1.select n_way classes randomly from all classes
    #         location = self.location.values.tolist()            
    #         rss = self.rss.values.tolist()
    #         rss_test = self.rss_test.values.tolist() 
    #         #print(np.array(rss).shape)
    #         from sklearn import neighbors
    #         knn = neighbors.KNeighborsRegressor(self.n_way-1, weights='uniform', metric='euclidean')
    #         neigh = knn.fit(rss, location)
    #         #random.seed(0)
    #         rss_id=(np.random.randint(90, size = 1))[0]
    #         distances, selected_cls = neigh.kneighbors(np.array(rss_test[rss_id]).reshape(-1,np.array(rss_test).shape[1]))
    #         selected_cls = (selected_cls[0]%90).tolist()
    #         selected_cls.append(rss_id)
            
    #         print('77777777777777')
    #         print(selected_cls)
    #         support_x = []
    #         query_x = []
 
    #         for cls in np.array(selected_cls):
    #             #print('888888888888888')
    #             #print(cls)
    #             # 2. select k_shot + k_query for each class
    #             #random.seed(0)
    #             #print('999999999999999999')
    #             #print(len(self.data))
    #             selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
    #             np.random.shuffle(selected_imgs_idx)
    #             indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
    #             indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
    #             support_x.append(
    #                 np.array(self.data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
    #             query_x.append(np.array(self.data[cls])[indexDtest].tolist())

    #         # shuffle the correponding relation between support set and query set
    #         #random.shuffle(support_x)
    #         #random.shuffle(query_x)

    #         self.support_x_batch.append(support_x)  # append set to current sets
    #         self.query_x_batch.append(query_x)  # append sets to current sets
            
    def calculate(self,image1, image2):
        # 灰度直方图算法
        # 计算单通道的直方图的相似值
        hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
        hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
        # 计算直方图的重合度
        degree = 0
        for i in range(len(hist1)):
            if hist1[i] != hist2[i]:
                degree = degree + \
                    (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
            degree = degree / len(hist1)
        return degree
        
    def classify_hist_with_split(self,image1,image2):
        # RGB每个通道的直方图相似度
        # 将图像resize后，分离为RGB三个通道，再计算每个通道的相似值
        # image1 = cv2.resize(image1, size)
        # image2 = cv2.resize(image2, size)
        sub_image1 = cv2.split(image1)
        sub_image2 = cv2.split(image2)
        sub_data = 0
        for im1, im2 in zip(sub_image1, sub_image2):
            sub_data += self.calculate(im1, im2)
        sub_data = sub_data / 3
        return sub_data

    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly from all classes
            location = self.location.values.tolist()            
            #print(np.array(rss).shape)
            #csi_id = (np.random.randint(90, size = 1))[0]
            csi_id = b
            with open('/home/student2/jgao/CSI/MetaLoc/CSIimage'+self.date+'/csi_test.csv', 'r') as f:
                reader = list(csv.reader(f))  
                index = -1
                for row in reader:
                    index = index + 1
                    if index == csi_id:
                        #print(row[0])
                        csi_test = cv2.imread(row[0])
            sim_list=[]
            with open('/home/student2/jgao/CSI/MetaLoc/CSIimage'+self.date+'/csi.csv', 'r') as f:
                reader = list(csv.reader(f))
                for row in reader:
                    csi = cv2.imread(row[0])
                    #print('999999999999999999')
                    #print(row[0])
                    n = self.classify_hist_with_split(csi, csi_test)
                    sim_list.append(n)            
            # distances, selected_cls = neigh.kneighbors(np.array(csi_test[csi_id]).reshape(-1,np.array(csi_test).shape[1]))
            
            # for i in range(csi):
            #     csi_test = cv2.imread(csi_test)
            #     csi = cv2.imread(csi)
            #     n = classify_hist_with_split(csi, csi_test[csi_id])
            #     sim_list.append(n) 
            #print('77777777777777')
            #selected_cls = sim_list.sort()
            #print(sim_list)
            sorted_id = sorted(range(len(sim_list)), key=lambda k: sim_list[k], reverse=True)
            #print(sorted_id)
            #print(csi_id)
            selected_cls = sorted_id[:10]
            selected_cls.append(csi_id)
            print('77777777777777')
            print(selected_cls)
            

            #selected_cls.append(csi_id)
            
            
            #print(selected_cls)
            support_x = []
            query_x = []
 
            for cls in np.array(selected_cls):
                #print('888888888888888')
                #print(cls)
                # 2. select k_shot + k_query for each class
                #random.seed(0)
                #print('999999999999999999')
                #print(len(self.data))
                selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())

            # shuffle the correponding relation between support set and query set
            #random.shuffle(support_x)
            #random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets


    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        # [setsz, 3, resize, resize]
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        # [setsz]
        support_y = np.zeros((self.setsz), dtype=np.int)
        # [querysz, 3, resize, resize]
        query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
        # [querysz]
        query_y = np.zeros((self.querysz), dtype=np.int)

        flatten_support_x = [os.path.join(self.path, item)
                             for sublist in self.support_x_batch[index] for item in sublist]
        

        support_y = np.array(
            [item.split('_')[0]  # filename:n0153282900000005.jpg, the first 9 characters treated as label
             for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)
#         print(self.support_x_batch[index])
#         print(support_y)
#         print('222222222222222222')
        flatten_query_x = [os.path.join(self.path, item)
                           for sublist in self.query_x_batch[index] for item in sublist]
        query_y = np.array([item.split('_')[0]
                            for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)

        # print('global:', support_y, query_y)
        # support_y: [setsz]
        # query_y: [querysz]
        # unique: [n-way], sorted
        unique = np.unique(support_y)
        random.shuffle(unique)
        # relative means the label ranges from 0 to n-way
        '''
        support_y_relative = np.zeros(self.setsz)
        query_y_relative = np.zeros(self.querysz)
        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx
            query_y_relative[query_y == l] = idx'''

        # print('relative:', support_y_relative, query_y_relative)

        for i, path in enumerate(flatten_support_x):
            support_x[i] = self.transform(path)
            
        for i, path in enumerate(flatten_query_x):
            query_x[i] = self.transform(path)

        return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz