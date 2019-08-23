import torch as th
import torch.utils.data as data
import cv2
import numpy as np
import os
import glob
import random
th.cuda.manual_seed(29)
th.manual_seed(29)
np.random.seed(0)
random.seed(0)

def loadImage(path):
#     print(path)
    inImage = cv2.imread(path, 0)
    info = np.iinfo(inImage.dtype) #(min = 0 ,max = 255),Maximum value of given dtype.
    inImage = inImage.astype(np.float) / info.max # 歸一化
#     inImage = inImage.astype(np.float) / 127.5-1 # 歸一化

    iw = inImage.shape[1]
    ih = inImage.shape[0]
    if iw <= ih:
        inImage = cv2.resize(inImage, (64, int(64 * ih/iw)))
#         print(inImage.shape)
    else:
        inImage = cv2.resize(inImage, (int(64 * iw / ih), 64)) #(160,80)->(128,64)
    inImage = inImage[0:64, 0:64]

#     inImage = cv2.resize(inImage, (48, 160))
    img = th.from_numpy(2 * inImage - 1).unsqueeze(0) # unsqueeze(0) 在第0維多增加一維 代表灰階，且輸入為-1~1之間
    # img shape = [1,64,64]  
#     print(th.max(img),th.min(img),th.mean(img))
    return img


class CASIABDataset(data.Dataset):
    def __init__(self, data_dir, target):
        self.data_dir = data_dir
        self.ids = np.arange(1, 63) #1-62
        self.cond = ['bg-01', 'bg-02', 'cl-01', 'cl-02',
                     'nm-01', 'nm-02', 'nm-03', 'nm-04',
                     'nm-05', 'nm-06']
#         self.angles = ['000', '018', '036', '054', '072',
#                        '108', '126', '144', '162', '180']  # originally
        self.angles = ['000', '018', '036', '054', '072', '090',
                       '108', '126', '144', '162', '180']
        self.n_id = 62
        self.n_cond = len(self.cond)
        self.n_ang = len(self.angles)
        print("n_con=",self.n_cond,',n_ang=',self.n_ang)
        self.target = target
        print('target = ',self.target)
        

    def __getitem__(self, index):
            # r1 is GT target
            # r2 is irrelevant GT target
            # r3 is source image   
        while(True):           
            id1 = th.randint(0, self. n_id, (1,)).item() + 1
            id1 = '%03d' % id1
            cond1 = th.randint(4, self.n_cond, (1,)).item()
            cond1 = int(cond1)
            cond1 = self.cond[cond1]
            r1 = id1 + '/' + cond1 + '/' + id1 + '-' + cond1 + '-' + self.target+'.png'
#             print("inside1:",r1) 
            if os.path.exists(self.data_dir + r1):
                break
#         print('r1=,',r1)


        id2 = id1 
        while(True):
            id2 = th.randint(0, self. n_id, (1,)).item() + 1
            id2 = '%03d' % id2
            cond2 = th.randint(4, self.n_cond, (1,)).item()
            cond2 = int(cond2)
            cond2 = self.cond[cond2]
            r2 = id2 + '/' + cond2 + '/' +  id2 + '-' + cond2 + '-' + self.target+'.png'
#             print("inside2:",r2)
            if os.path.exists(self.data_dir + r2) and (id2!=id1):
                break
#         print('r2=,',r2)

        while True:
            angle = th.randint(0, self.n_ang, (1,)).item()
            angle = int(angle)
            angle = self.angles[angle]
            cond3 = th.randint(0, self.n_cond, (1,)).item()
            cond3 = int(cond3)
            cond3 = self.cond[cond3]
            r3 = id1 + '/' + cond3 + '/'  +  id1 + '-' + cond3 + '-' + angle + '.png'
            if os.path.exists(self.data_dir + r3):
                break
#         print('r3=,',r3,'\n')

        img1 = loadImage(self.data_dir + r1)
        img2 = loadImage(self.data_dir + r2)
        img3 = loadImage(self.data_dir + r3)
        return img1, img2, img3
    
    def __len__(self):
        total_len = 0
        total_len = len(glob.glob(self.data_dir))
        return 6400
#         return th.stack(batch1), th.stack(batch2), th.stack(batch3)
    

# class CASIABDatasetForTest():
#     def __init__(self, data_dir):
#         self.data_dir = data_dir
#         self.ids = np.arange(63, 125)
#         self.cond = ['bg-01', 'bg-02', 'cl-01', 'cl-02',
#                      'nm-01', 'nm-02', 'nm-03', 'nm-04',
#                      'nm-05', 'nm-06']
#         self.angles = ['000', '018', '036', '054', '072',
#                        '108', '126', '144', '162', '180']
#         self.n_id = 62
#         self.n_cond = len(self.cond)
#         self.n_ang = len(self.angles)

#     def getbatch(self, batchsize):
#         batch1 = []
#         batch2 = []
#         batch3 = []
#         for i in range(batchsize):
#             seed = th.randint(1, 100000, (1,)).item()
#             print('test_dataset seed',seed)
#             th.manual_seed((i+1)*seed)
#             # r1 is GT target
#             # r2 is irrelevant GT target
#             # r3 is source image
#             id1 = th.randint(0, self. n_id, (1,)).item() + 1
#             id1 = '%03d' % id1
#             # cond1 = th.randint(4, self.n_cond, (1,)).item()
#             # cond1 = int(cond1)
#             # cond1 = self.cond[cond1]
#             cond1 = 'nm-01'
#             r1 = id1 + '/' + cond1 + '/' + id1 + '-' + \
#                 cond1 + '-' + '090.png'

#             id2 = id1
#             while (id2 == id1):
#                 id2 = th.randint(0, self. n_id, (1,)).item() + 1
#                 id2 = '%03d' % id2
#                 # cond2 = th.randint(4, self.n_cond, (1,)).item()
#                 # cond2 = int(cond2)
#                 # cond2 = self.cond[cond2]
#                 cond2 = 'nm-01'
#                 r2 = id2 + '/' + cond2 + '/' + id2 + '-' + \
#                     cond2 + '-' + '090.png'
#             while True:
#                 angle = th.randint(0, self.n_ang, (1,)).item()
#                 angle = int(angle)
#                 angle = self.angles[angle]
#                 cond3 = th.randint(0, self.n_cond, (1,)).item()
#                 cond3 = int(cond3)
#                 cond3 = self.cond[cond3]

#                 r3 = id1 + '/' + cond3 + '/' + id1 + '-' + \
#                     cond3 + '-' + angle + '.png'
#                 if os.path.exists(self.data_dir + r3):
#                     break

#             img1 = loadImage(self.data_dir + r1)
#             img2 = loadImage(self.data_dir + r2)
#             img3 = loadImage(self.data_dir + r3)
#             batch1.append(img1)
#             batch2.append(img2)
#             batch3.append(img3)
#         return th.stack(batch1), th.stack(batch2), th.stack(batch3)

class CASIABDatasetForTest():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.ids = np.arange(63, 125)
        self.cond = ['bg-01', 'bg-02', 'cl-01', 'cl-02',
                     'nm-01', 'nm-02', 'nm-03', 'nm-04',
                     'nm-05', 'nm-06']
        self.angles = ['000', '018', '036', '054', '072', '090'
                       '108', '126', '144', '162', '180']
        self.n_id = 62
        self.n_cond = len(self.cond)
        self.n_ang = len(self.angles)

    def getbatch(self, batchsize):
        batch1 = []
        batch2 = []
        batch3 = []
        for i in range(batchsize):
            
            # r1 is GT target
            # r2 is irrelevant GT target
            # r3 is source image

            id1 = th.randint(62, 62+self.n_id, (1,)).item() + 1
            id1 = '%03d' % id1
            while(True):
                cond1 = th.randint(4, self.n_cond, (1,)).item()
                cond1 = int(cond1)
                cond1 = self.cond[cond1]
    #           cond1 = 'nm-01'
                r1 = id1 + '/' + cond1 + '/' + id1 + '-' + cond1 + '-' + '090.png'
                if os.path.exists(self.data_dir + r1):
                    break
#             print('r1,',r1)

            id2 = id1 
            while (id2 == id1):
                id2 = th.randint(62, 62+self.n_id, (1,)).item() + 1
                id2 = '%03d' % id2
            while(True):  
                cond2 = th.randint(4, self.n_cond, (1,)).item()
                cond2 = int(cond2)
                cond2 = self.cond[cond2]
    #           cond2 = 'nm-01'
                r2 = id2 + '/' + cond2 + '/' +  id2 + '-' + cond2 + '-' + '090.png'
                if os.path.exists(self.data_dir + r2):
                    break
#             print('r2,',r2)
    
            while True:
                angle = th.randint(0, self.n_ang, (1,)).item()
                angle = int(angle)
                angle = self.angles[angle]
                cond3 = th.randint(0, self.n_cond, (1,)).item()
                cond3 = int(cond3)
                cond3 = self.cond[cond3]
                r3 = id1 + '/' + cond3 + '/'  +  id1 + '-' + cond3 + '-' + angle + '.png'
                if os.path.exists(self.data_dir + r3):
                    break
#             print('r3,',r3,'\n')
                    
            img1 = loadImage(self.data_dir + r1)
            img2 = loadImage(self.data_dir + r2)
            img3 = loadImage(self.data_dir + r3)
            batch1.append(img1)
            batch2.append(img2)
            batch3.append(img3)
        return th.stack(batch1), th.stack(batch2), th.stack(batch3)    

class CASIABDatasetGenerate():
    def __init__(self, data_dir, cond):
        self.data_dir = data_dir
        self.ids = np.arange(63, 125)
        self.angles = ['000', '018', '036', '054', '072', '090',
                       '108', '126', '144', '162', '180']
        self.n_ang = len(self.angles)
        self.cond = cond

    def getbatch(self, idx, batchsize):
        batch1 = []
        batch3 = []
        id1 = idx
        id1 = '%03d' % id1
        cond1 = self.cond
#         r1 = id1 + '/' + cond1 + '/' + '090' + '/' + id1 + '-' + \
#             cond1 + '-' + '090.png'
        r1 = id1 + '/' + cond1 + '/' + id1 + '-' + cond1 + '-' + '090.png'
        if not os.path.exists(self.data_dir + r1):
            img1 = th.from_numpy(np.zeros((64, 64))).unsqueeze(0)
#                 img1 = th.from_numpy(np.zeros((160, 48))).unsqueeze(0)
        else:
            img1 = loadImage(self.data_dir + r1)
        for angle in self.angles:
            # r1 is GT target
            # r2 is source image
#             r3 = id1 + '/' + cond1 + '/' + angle + '/' + id1 + '-' + \
#                 cond1 + '-' + angle + '.png'
            r3 = id1 + '/' + cond1 + '/' + id1 + '-' + cond1 + '-' + angle + '.png'
            if not os.path.exists(self.data_dir + r3):
                img3 = th.from_numpy(np.zeros((64, 64))).unsqueeze(0)
#                 img3 = th.from_numpy(np.zeros((160, 48))).unsqueeze(0)
            else:
                img3 = loadImage(self.data_dir + r3) #1,64,64
            
           
            batch1.append(img1)
            batch3.append(img3)
#         return  th.stack(batch3)
        return  th.stack(batch1), th.stack(batch3)  #11,1,64,64
