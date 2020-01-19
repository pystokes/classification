#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

def train(loader_train, model_obj, optimizer, loss_fn, device, total_epoch, epoch):
    
    model_obj.train() # モデルを学習モードに変更
 
    # ミニバッチごとに学習
    for data, targets in loader_train:
 
        # データを計算に使用するデバイスに転送するため，to()で明示的に指定
        data = data.to(device)
        targets = targets.to(device)
 
        optimizer.zero_grad() # 勾配を初期化
        outputs = model_obj(data, mode='train') # 順伝播の計算
        loss = loss_fn(outputs, targets.long()) # 損失を計算
 
        loss.backward() # 誤差を逆伝播させる
        optimizer.step() # 重みを更新する
    
    return loss


def test(loader_test, trained_model, device):
 
    trained_model.eval() # モデルを推論モードに変更
    correct = 0 # 正当数のカウント用変数
 
    # ミニバッチごとに推論
    with torch.no_grad(): # 推論時には勾配は不要
        for data, targets in loader_test:
 
            # データを計算に使用するデバイスに転送するため，to()で明示的に指定
            data = data.to(device)
            targets = targets.to(device)
 
            # 順伝播の計算
            outputs = trained_model(data, mode='test')
 
            # 推論結果から確率が最大のインデックスを取得
            # torch.max()の返り値：
            #   - 第一返り値：値（values）
            #   - 第二返り値：インデックス（indices）
            _, predicted = torch.max(outputs.data, dim=1)
            # 正解数をカウントアップ
            correct += predicted.eq(targets.long().data.view_as(predicted)).sum()
    
    # テストデータの総数
    data_num = len(loader_test.dataset) 

    return correct, data_num
