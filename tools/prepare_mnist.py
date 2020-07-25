#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import cv2
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def create_mnist_data(save_path='../mnist_data'):

    # MNISTのデータセットを取得（初回実行時はダウンロードするため時間がかかる）
    print('\nLoading MNIST dataset in:', save_path)
    mnist = fetch_openml('mnist_784', data_home=save_path)
    x = mnist.data
    y = mnist.target

    print('\n--- Shape of original data -------------------')
    print('x:', x.shape)
    print('y:', y.shape)
    print('----------------------------------------------\n')

    # データ形式を変換
    # 変換前：[画像数, 784]（784=28x28）
    # 変換後：[画像数, 28, 28]（グレースケールでありチャネル数は1のため省略，高さと幅はMNISTは28）
    num_x = len(x)
    x = x.reshape(num_x, 28, 28)

    # データを学習用とテスト用に分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    print('--- Shape of splitted data -------------------')
    print('x_train:', x_train.shape)
    print(' x_test:', x_test.shape)
    print('y_train:', y_train.shape)
    print(' y_test:', y_test.shape)
    print('----------------------------------------------\n')
 
    # 画像として保存
    save_dir = Path(save_path)
    for data_type in ['train', 'test']:

        if data_type == 'train':
            x_data = x_train
            y_data = y_train
        else:
            x_data = x_test
            y_data = y_test
        
        for idx in range(len(x_data)):

            if idx % 1000 == 0:
                print(f'Progress: {data_type} {idx:05}/{len(x_data)}')

            img = x_data[idx]
            label = y_data[idx]

            each_save_dir = save_dir.joinpath(data_type, label)
            each_save_dir.mkdir(exist_ok=True, parents=True)
            
            each_save_path = each_save_dir.joinpath(f'{idx:08}.jpg')
            cv2.imwrite(str(each_save_path), img)

if __name__ == '__main__':

    create_mnist_data()
