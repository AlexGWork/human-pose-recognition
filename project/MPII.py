# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:18:17 2019

@author: Aleksandr
"""

import pandas as pd
import os
import glob
from PIL import Image
import torchvision.transforms as transforms
from torch.utils import data as D

class MPII(D.Dataset):
    def __init__(self, imgs_root, csv_path, is_train=True):
        
        """
        Класс Dataset под данные MPII
        imgs_root: путь к папке с изображениями
        csv_path:  путь к csv, где всего две колонки без заголовочной строки - 
                   имя и метка класса
        is_train:  обучающая выборка или нет
        """
        self.is_train = is_train
        # полные пути до файлов
        self.filenames = []
        # трансформации изображений
        self.transform = transforms.ToTensor()
        # название файлов, для сопоставления с csv файлом
        short_filenames = []
        # исходные метки изображений
        self.labels = []      
        """
        Добавляем все картинки из директории в filenames.
        Если выборка обучающая, то сопоставляем названия
        картинок с лейблами из csv
        """
        if is_train:
            filenames = glob.glob(os.path.join(imgs_root, '*.jpg'))  
            for file in filenames:
                self.filenames.append(file)
                _, tail = os.path.split(file)
                short_filenames.append(tail)
            csv_df = pd.read_csv(csv_path, names = ['name', 'label'])
            str_num = csv_df['name'].size
            for file in short_filenames:
                for i in range(0, str_num):
                    if file == csv_df['name'][i]:
                        self.labels.append(int(csv_df['label'][i]))
        else:
            self.filenames.append(imgs_root)
    
    def __getitem__(self, index):
        image = Image.open(self.filenames[index])
        if self.is_train:
            return self.transform(image), self.labels[index]
        else:
            return self.transform(image)
    
    def __len__(self):
        return len(self.filenames)