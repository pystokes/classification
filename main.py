#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime as dt
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch import optim
from data_loader import create_mnist_data_loader
from model import CNN
from trainer import train, test

def  main(config):
 
    # 使用リソースの設定（CPU or GPU）
    # PyTorchでは明示的に指定する必要がある
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\nDevice:', device)

    # Data Loaderの作成
    train_loader, test_loader = create_mnist_data_loader(config)

    # モデル作成
    model = CNN(config).to(device)
    print(model) # ネットワークの詳細の確認用に表示
 
    # 損失関数を定義
    # CorssEntropyLoss = LogSoftmax + NLLLossであるため，学習時はモデルからの出力は確率化しない
    loss_fn = nn.CrossEntropyLoss()
 
    # 最適化手法を定義（ここでは例としてAdamを選択）
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
 
    # 9. 学習（各エポックごとにテスト用データで精度：Accを計算）
    print('\n-----------------')
    print('   Begin train')
    print('-----------------\n')
    for epoch in range(1, config['train']['epochs']+1):
        loss = train(train_loader, model, optimizer, loss_fn, device, config['train']['epochs'], epoch)
        correct, data_num = test(test_loader, model, device)
        print(f'{dt.datetime.now().strftime("%H:%M:%S")} Epoch [{epoch:05}/{config["train"]["epochs"]:05}], Train Loss: {loss.item():.4f}, Test Acc: {correct:05}/{data_num:05} ({(100. * float(correct) /data_num):.1f}%)')

        # save_periodごとに学習した重みを保存
        if epoch % config['train']['save_period'] == 0:
            # 保存先ディレクトリを作成
            save_home = Path(config['train']['save_home'])
            save_home.mkdir(exist_ok=True, parents=True)
            # 保存するファイルのパスを設定
            save_path = save_home.joinpath(f'weight-{str(epoch).zfill(5)}.pth')
            # 重みを保存
            torch.save(model.state_dict(), save_path)

if __name__ == '__main__':

    from pprint import pprint

    with open('./config.json', 'r') as f:
        config = json.load(f)
    
    print('\n--- Config ---------------')
    pprint(config)
    print('--------------------------')

    main(config)
