import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from PIL import Image

# # 读取文件
# class Data_process():
#     def __init__(self):
#         self.train_path = glob('./all_data/*')
#
#     def get_file_name(self):

# 将CT影像保存到train文件夹中
# for step1, file in enumerate(glob('.\\all_data\\*')):
#     path1 = './train/'+str(1001+step1)
#     if not os.path.exists(path1):
#         os.makedirs(path1)
#     for step2, filename in enumerate(glob(file+'\\'+'arterial phase\\*')):
#         img = Image.open(filename)
#         img.save(path1+'/'+str(step2+1)+'.png')

# 将掩膜保存到label文件夹中
for step1, file in enumerate(glob('.\\数据集1\\*')):
    path1 = './label/'+str(1001+step1)
    if not os.path.exists(path1):
        os.makedirs(path1)
    for step2, filename in enumerate(glob(file+'\\'+'arterial phase\\*.png')):
        img = Image.open(filename)
        img.save(path1+'/mask_'+str(step2+1)+'.png')