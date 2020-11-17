# -*- coding: utf-8 -*-
"""
@author: Aleksandr
"""
import torch
import convnet
import MPII
from torch.utils import data as D

width = 600
height = 480
chanels = 3

dtype = torch.float32
USE_GPU = True
PATH = 'model.pth'
labels_dict = {0: 'ходьба',
               1: 'бег',
               2: 'сидит',
               3: 'стоит',
               4: 'пилатес'}
#параметры инициализации
num_filters = 9 # depth
conv_kernel = 3 # 3*3
pooling_kernel = 2 #2*2
linear1 = 10
linear2 = 5    

def process(img_path):
    device = 0
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('using device:', device)    
	# инициализация
    im_size = (width, height, chanels)    
    conv_tuple =  (num_filters, conv_kernel, pooling_kernel)
    conv_params = [conv_tuple, conv_tuple, conv_tuple]
    linear_params = [linear1, linear2]
    model = convnet.ConvNet(im_size, conv_params, linear_params)
    model.load_state_dict(torch.load(PATH))    
    model.eval() # отключение обучения, будет просто метки выдавать
    model = model.to(device=device)
    test_dataset = MPII.MPII(imgs_root=img_path, csv_path=' ', is_train=False) #прослойка для считывания данных
    test_dataloader = D.DataLoader(test_dataset, batch_size = 1) # загрузчик данных
    dataiter = iter(test_dataloader)
    images = dataiter.next()
    images = images.to(device=device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    label = labels_dict[predicted[0].item()]
    return label
