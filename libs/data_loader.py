#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def create_mnist_data_loader(config):

    # MNISTのデータセットを取得（初回実行時はダウンロードするため時間がかかる）
    print('\nLoading MNIST dataset in:', config['preprocess']['data_home'])
    mnist = fetch_openml('mnist_784', data_home=config['preprocess']['data_home'])

    # 入出力データの設定（入力データは255で割ることで[0, 1]に正規化する）
    # 注意：yはintではなくてstringで格納されているため，intに変換する必要がある
    x = mnist.data / 255
    y = np.array([*map(int, mnist.target)]) # mapで一括変換

    print('\n--- Shape of original data -------------------')
    print('x:', x.shape)
    print('y:', y.shape)
    print('----------------------------------------------\n')

    # データ形式を変換：PyTorchでの形式＝[画像数，チャネル数，高さ，幅]
    # 変換前：[画像数, 784]（784=28x28）
    # 変換後：[画像数, 1, 28, 28]（グレースケールなのでチャネル数は1，高さと幅はMNISTは28）
    num_x = len(x)
    x = x.reshape(num_x, 1, 28, 28)

    # データを学習用とテスト用に分割
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=config['preprocess']['test_size'],
                                                        random_state=config['preprocess']['random_state'])
    print('--- Shape of splitted data -------------------')
    print('x_train:', x_train.shape)
    print(' x_test:', x_test.shape)
    print('y_train:', y_train.shape)
    print(' y_test:', y_test.shape)
    print('----------------------------------------------\n')
 
    # PyTorchのテンソルに変換
    x_train = torch.from_numpy(x_train.astype(np.float32))
    x_test = torch.from_numpy(x_test.astype(np.float32))
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)
 
    # 入力（x）とラベル（y）を組み合わせて最終的なデータを作成
    ds_train = TensorDataset(x_train, y_train)
    ds_test = TensorDataset(x_test, y_test)
 
    # DataLoaderを作成
    loader_train = DataLoader(ds_train,
                              batch_size=config['train']['batch_size'],
                              shuffle=config['train']['shuffle'])
    loader_test = DataLoader(ds_test,
                              batch_size=config['train']['batch_size'],
                              shuffle=False)
 
    return loader_train, loader_test
