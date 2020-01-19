#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

def train(loader_train, model_obj, optimizer, loss_fn, device, total_epoch, epoch):
    
    model_obj.train() # モデルを学習モードに変更
 
    # ミニバッチごとに学習
    for data, targets in loader_train:
 
        data = data.to(device) # GPUを使用するため，to()で明示的に指定
        targets = targets.to(device) # 同上
 
        optimizer.zero_grad() # 勾配を初期化
        outputs = model_obj(data, mode='train') # 順伝播の計算
        loss = loss_fn(outputs, targets.long()) # 誤差を計算
 
        loss.backward() # 誤差を逆伝播させる
        optimizer.step() # 重みを更新する
    
    return loss


def test(loader_test, trained_model, device):
 
    trained_model.eval() # モデルを推論モードに変更
    correct = 0 # 正当数のカウント用変数
 
    # ミニバッチごとに推論
    with torch.no_grad(): # 推論時には勾配は不要
        for data, targets in loader_test:
 
            data = data.to(device) #  GPUを使用するため，to()で明示的に指定
            targets = targets.to(device) # 同上
 
            outputs = trained_model(data, mode='test') # 順伝播の計算
 
            # 推論結果の取得と正誤判定
            _, predicted = torch.max(outputs.data, 1) # 確率が最大のラベルを取得
            correct += predicted.eq(targets.long().data.view_as(predicted)).sum() # 正解ならば正解数をカウントアップ
    
    # 正解率を計算
    data_num = len(loader_test.dataset) # テストデータの総数

    return correct, data_num
