import os
import random
import numpy as np
from tqdm import tqdm
from utils import *

train_percent = 1
val_percent = 0
test_percent = 0

data_path = 'data/PRD289K'

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt for trainning, validating and testing in data folder.")
    # segfilepath = os.path.join(data_path, "labels")
    segfilepath = r"/mnt/ImarsData/ljs/PRD289K"
    saveBasePath = os.path.join(data_path, "list")
    
    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".tif") or seg.endswith(".TIF"):
            total_seg.append(seg)

    num = len(total_seg)   
    train_num = int(num*train_percent)  
    val_num = int(num*val_percent)
    test_num = int(num*test_percent)

    print("Train size: {:d} | Val size: {:d} | Test size: {:d}".format(train_num, val_num, test_num))

    # shuffle the list
    random.shuffle(total_seg)

    train_list = total_seg[0:int(num*1)]
    val_list = total_seg[int(num*train_percent):int(num*(train_percent+val_percent))]
    # test_list = total_seg[int(num*(train_percent+val_percent)):int(num*(train_percent+val_percent+test_percent))]

    # create the list file for trainning, validating and testing in data folder
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w') 
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')  
    # ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')  
    
    for i in train_list:  
        # linux should add replace("\\", "\\\\")
        ftrain.write(os.path.join(segfilepath, i).replace("\\", "\\\\") + '\n')
        # ftrain.write(os.path.join(segfilepath, i) + '\n')
    for i in val_list:  
        # linux should add replace("\\", "\\\\")
        fval.write(os.path.join(segfilepath, i).replace("\\", "\\\\") + '\n')
    #     # fval.write(os.path.join(segfilepath, i) + '\n')
    # for i in test_list:  
    #     # linux should add replace("\\", "\\\\")
    #     ftest.write(os.path.join(segfilepath, i).replace("\\", "\\\\") + '\n')
    #     # ftest.write(os.path.join(segfilepath, i) + '\n')

    ftrain.close()  
    fval.close()  
    # ftest.close()
    print("Create list successfully.")