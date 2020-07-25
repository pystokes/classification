#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

def create_mnist_data_loader(config):

    # データセットのホームディレクトリを設定
    data_home = Path(config['data_home'])

    # 学習データのクラス名を取得


    # クラス数・ラベル名の読み込み

    # 画像Pathリストの作成

    # BatchDatasetの作成
 
    # DataLoaderを作成
    loader_train = DataLoader(xxx,
                              batch_size=config['train']['batch_size'],
                              shuffle=config['train']['shuffle'])
    loader_test = DataLoader(xxx,
                              batch_size=config['train']['batch_size'],
                              shuffle=False)
 
    return loader_train, loader_test
