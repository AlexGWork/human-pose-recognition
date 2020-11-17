# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 00:56:14 2019

@author: Aleksandr
"""

import os
import glob
from PIL import Image
import pandas as pd
from shutil import copy
 
def scale_image(input_image_path: str,
                output_image_path: str,
                width=None,
                height=None
                ):
    """
    сжимает картинку input_image_path
    до размеров (width, height)
    и размещает сжатое изображение в
    output_image_path
    """
    original_image = Image.open(input_image_path)
    w, h = original_image.size
    if width and height:
        max_size = (width, height)
    else:
        raise RuntimeError('Width or height required!')
    original_image = original_image.convert("RGB")
    original_image = original_image.resize(max_size, Image.LANCZOS)
    original_image.save(output_image_path)
    
def scale_images(input_dir:str, output_dir:str, height = 480, width = 600):
    """
    сжимает картинки из input_dir
    до размера (width, height)
    и сохраняет их в output_dir
    """
    all_files = glob.glob(os.path.join(input_dir, '*.jpg'))
    for file in all_files:
        _, tail = os.path.split(file)
        out = output_dir + tail
        scale_image(file, out, width, height)
        
def copy_files(source_dir: str, destination_dir: str, files: list):
    """
    Ищет в source_dir файлы из files и копирует в destination_dir.
    Возвращает список файлов, которые не были найдены в files
    """
    fails = []
    for file in files:
        try:
            copy(source_dir + file, destination_dir)
        except FileNotFoundError:
            fails.append(file)
    return fails

def copy_by_names(source_dir:str, out_dir:str, csv_path:str):
    """
    Метод копирует изображения из исходной директории, если их имена есть
    в csv-файле
    source_dir: Исходая директория
    out_dir: Выходная директория
    csv_path: Путь к csv-файлу, где первая колонка - имя файла, вторая - метка
    """
    df = pd.read_csv(csv_path, names = ['name', 'label'])   
    file_names = []
    name_column = df['name'].to_list()
    for name in name_column:
        if os.path.exists(source_dir+name):
            file_names.append(name)

    copy_files(source_dir, out_dir, file_names)  
    
def get_files_by_labels(csv_path:str, new_csv_path:str, labels_and_counts:list, 
                          imgs_source_dir:str, action = True):
    """
    Возвращает список файлов, соответствующих лейблам из labels_and_counts,
    причем на каждый лейбл не более counts файлов из labels_and_counts
    csv_path: путь до исходного csv
    new_csv_path: путь до нового csv
    labels_and_counts: список кортежей, где каждый кортеж (int, int)  это
    (label, count)
    imgs_source_dir: директория с картинками
    action: True - лейбл-действие, False - лейбл-категория    
    """  
    df = pd.read_csv(csv_path, names = ['name', 'train', 'action', 'category'])    
    raws = df.shape[0]    
    file_names = []    
    old_labels = []
    for label, base_count in labels_and_counts: 
        old_labels.append(label)
        count = base_count
        for i in range(1, raws):
            if count <= 0:
                break
            if os.path.exists(imgs_source_dir + df['name'][i]):
                if (action and int(df['action'][i]) == label) or (not action and int(df['category'][i]) == label):
                    count -= 1
                    file_names.append(df['name'][i])
    """
    переразметка лейблов
    """
    final_normalization(old_labels, True, csv_path, new_csv_path)
        
    return file_names

def base_label_normalization(csv_path: str, out_csv_path: str, action = True):    
    """
    Метод делает все лейблы в csv-файле числами, начиная с 0 с шагом 1
    csv_path: исходный путь до csv
    out_csv_path: выходной путь до csv
    action: True - метка это действие. False - метка это категория
    """ 
    csv_df = pd.read_csv(csv_path, names = ['name', 'train', 'action', 
                                                'category'])
    labels = {}
    label_counter = 0
    column = ''
    if action:
        column = 'action'
    else:
        column = 'category'
    for old_label in csv_df[column]:
        if old_label not in labels.keys() and old_label >= 0:
            labels[old_label] = label_counter
            label_counter += 1       
    raws = csv_df.shape[0]
    for i in range(raws):
        old_label = csv_df[column][i].item()
        csv_df[column][i] = labels.get(old_label, -1) 
    csv_df.to_csv(out_csv_path, index = False)
    
def final_normalization(old_labels: list, csv_path: str, out_csv_path: str,
                        action: bool):  
    """
    Метод удаляет лишние строки и столбцы, а такжже заменяет старые ЧИСЛОВЫЕ
    метки класса на новые ЧИСЛОВЫЕ метки, начинающиеся с 0.
    Все записи с меткой не из old_labels будут удалены.
    Останутся две колонки: файл и лейбл
    old_labels: list со старыми метками класса.
    csv_path: путь к исходному csv-файлу
    out_csv: путь к csv-файлу, куда будет сохранен датафрейм
    action: True - метки-действия, False - метки - категории
    """
    df = pd.read_csv(csv_path, names = ['name', 'train', 'action', 
                                                'category'])
    # удаление лишних столбцов
    column = 'action'
    if(action):
        df.drop(['train','category'], axis = 1, inplace =True)
    else:
        df.drop(['train','action'], axis = 1, inplace =True)
        column = 'category'
    # формирование новых меток
    new_labels = {}
    label_counter = 0
    for label in old_labels:
        new_labels[label] = label_counter
        label_counter += 1
    # удаление лишних строк
    j = 1
    rows = df.shape[0]
    while j < rows:
        if int(df.loc[j, column]) not in new_labels.keys():
            df.drop(j, inplace = True)
        else:
            df.loc[j, column] = new_labels.get(int(df.loc[j, column]), -1)
        j += 1
    df.to_csv(out_csv_path, header = False, index = False)
    
"""НОВЫЙ МЕТОД"""

def create_experiment(categories: list, out_csv_path: str,
                               csv_path: str, is_action: bool):
    
    """ 
    Я еще не проверял этот метод, если is_action=True
    Метод подготовки csv-файла для эксперимента с метками-категориями
    categories: список строк с названиями категорий, по которым
                будет проводиться эксперимент
    out_csv_path: путь к новому csv файлу
    csv_path: путь к старому csv файлу
    is_action: True - лейблы в столбце action, False - лейблы в столбце category
    """
    df = pd.read_csv(csv_path, names = ['name', 'train', 'action', 
                                                'category'])
    """
    Предварительное удаление лишних столбцов и строк не из тестовой выборки
    """
    df_remove_test = df[df['train']==0]
    df.drop(df_remove_test.index, axis =0, inplace = True)
    if not is_action:
        df.drop(['train','action'], axis =1, inplace =True)
        df = df.query('category in @categories')
    else:
        df.drop(['train','category'], axis =1, inplace =True)
        df = df.query('action in @categories')
    """
    Замена категорий/действий на числовые лейблы, начиная с 0 и без пропусков
    """
    new_labels = {}
    label_counter = 0
    for category in categories:
        new_labels[category] = label_counter
        label_counter += 1
    df = df.replace(new_labels)
    df.to_csv(out_csv_path, header = False, index = False)  

#categories = ['fishing and hunting', 'lawn and garden', 'water activities', 
#              'dancing', 'music playing']
#out_csv_path = 'D:\_github\experiments\experient_3\\5_categories.csv'
#csv_path = 'D:\_github\experiments\mpii9.csv'

##create_experiment(categories, out_csv_path, csv_path, False)

#source_dir1 = 'D:\_github\datasets\data\cut_Images1\\'
#out_dir1 = 'D:\_github\experiments\experient_3\imgs1\\'
#source_dir2 = 'D:\_github\datasets\data\cut_Images2\\'
#out_dir2 = 'D:\_github\experiments\experient_3\imgs2\\'
#source_dir3 = 'D:\_github\datasets\data\cut_Images3\\'
#out_dir3 = 'D:\_github\experiments\experient_3\imgs3\\'
#copy_by_names(source_dir1, out_dir1, out_csv_path)
#copy_by_names(source_dir2, out_dir2, out_csv_path)
#copy_by_names(source_dir3, out_dir3, out_csv_path)
