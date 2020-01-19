#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module): # nn.Moduleを継承する
 
    def __init__(self, config):
 
        super(CNN, self).__init__()

        # 入力サイズ: バッチサイズ=N, チャネル=1, 高さ=28, 幅=28

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Feature mapのサイズを半分にする
            nn.BatchNorm2d(16)
        ) # 出力サイズ: バッチサイズ=N, チャネル=16, 高さ=14, 幅=14

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Feature mapのサイズを半分にする
            nn.BatchNorm2d(32)
        ) # 出力サイズ: バッチサイズ=N, チャネル=32, 高さ=7, 幅=7

        self.full_connection = nn.Sequential(
            nn.Linear(in_features=32*7*7, out_features=512), # in_featuresは直前の出力ユニット数（チャネル x 高さ x 幅）
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=config['train']['num_classes'])
        )

 
    # Forward計算の定義
    # 参考：Define by Runの特徴（任意の条件で処理を動的に変更できる）
    def forward(self, x, mode='train'):
 
        x = self.block1(x)
        x = self.block2(x)
 
        # 直前の出力が2次元（× チャネル数）なので，全結合（1次元）の入力形式に変換
        #
        # 変換前: [バッチ数, チャネル数, 高さ, 幅] (4次元)
        # 変換後: [バッチ数, チャネル数 x 高さ x 幅] (2次元)
        #
        # view()：引数で指定したサイズのベクトルに形を変換する
        #   x.size(0): バッチサイズ
        #          -1: -1を指定することで，「残りの次元（C, H, W）をすべてまとめて1次元に変換する」の意味になる
        #
        # 参考：KerasのFlatten()と同じような処理
        x = x.view(x.size(0), -1)
 
        # 最終的な出力は確率になっていないことに注意
        x = self.full_connection(x)

        # Testのときのみ出力を確率に変換する
        # 注意：TrainのときはLossにLogSoftmaxの処理が含まれているため変換不要
        if mode == 'test':
            x = F.softmax(x, dim=1) # xは[N, num_classes]形式であるため，dim=1方向で確率化する

        return x
