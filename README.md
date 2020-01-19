# MNIST

MNISTの手書き文字分類問題をPyTorchで実装したサンプルコード。

## 環境情報

以下の環境で実装したものですが，各種バージョンへの依存性はそれ程強くはないため，比較的新しいバージョンであれば動作すると思います。

- Windows 10
- Python 3.6.4
- CUDA 9.2
- pip

CUDAの詳細は以下の通り。

```bash
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Tue_Jun_12_23:08:12_Central_Daylight_Time_2018
Cuda compilation tools, release 9.2, V9.2.148
```

## インストール方法

### 前提条件

PythonおよびCUDAのインストールが完了していること。

### ライブラリのインストール

#### [環境情報](#環境情報)を満たしている場合

このリポジトリを任意のディレクトリにクローンする。

```bash
git clone https://github.com/pystokes/mnist.git
```

クローンしたリポジトリに移動し，必要なライブラリをインストールして完了。

```bash
cd mnist
pip install -r requirements.txt
```

#### [環境情報](#環境情報)を満たしていない場合

1. PyTorchのインストール
    
    [公式サイト](https://pytorch.org/get-started/locally/)から環境にあった方法でインストールする。

    例：上記の環境情報を満たしている場合は以下のようになる。
    ```bash
    pip3 install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
    ```

2. scikit-learnのインストール

    ```bash
    pip install sklearn
    ```

3. このリポジトリを任意のディレクトリにクローンして完了。

    ```bash
    git clone https://github.com/pystokes/mnist.git
    ```

## 使い方

1. [config.json](./config.json)の各種パラメータを設定する。

    ```bash
    {
        # データの前処理に関するパラメータ
        "preprocess": {
            # ダウンロードするMNISTデータセットの保存場所
            "data_home": "./mnist_data",
            # データを学習/テストに分割する際の乱数を制御するパラメータ
            "random_state": 0,
            # データを分割する際のテストデータの割合
            "test_size": 0.3
        },
        # 学習時のパラメータ
        "train": {
            # 分類するクラス数（MNISTは0~9の10クラス分類）
            "num_classes": 10,
            # 学習/推論時のバッチサイズ
            "batch_size": 128,
            # 最適化する際の学習率
            "learning_rate": 0.001,
            # DataLoaderでデータをシャッフルするか否かを決めるパラメータ
            "shuffle": true,
            # トータルのエポック数（学習する回数）
            "epochs": 10,
            # 結果（学習済みモデルの重み）の保存先
            "save_home": "./results",
            # 何エポックごとに結果（学習済みモデルの重み）を保存するかを指定
            "save_period": 2
        }
    }
    ```

2. 学習を実行。

    ```bash
    python main.py
    ```

    上記のコマンドを実行すると以下のようなログが標準出力に表示される。

    ```bash
    python main.py

    --- Config ---------------
    {'preprocess': {'data_home': './mnist_data',
                    'random_state': 0,
                    'test_size': 0.3},
    'train': {'batch_size': 128,
            'epochs': 10,
            'learning_rate': 0.001,
            'num_classes': 10,
            'save_home': './results',
            'save_period': 2,
            'shuffle': True}}
    --------------------------

    Device: cuda

    Loading MNIST dataset in: ./mnist_data
    Loaded

    --- Shape of original data -------------------
    x: (70000, 784)
    y: (70000,)
    ----------------------------------------------

    --- Shape of splitted data -------------------
    x_train: (49000, 1, 28, 28)
     x_test: (21000, 1, 28, 28)
    y_train: (49000,)
     y_test: (21000,)
    ----------------------------------------------

    CNN(
      (block1): Sequential(
        (0): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace=True)
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (block2): Sequential(
        (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace=True)
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (full_connection): Sequential(
        (0): Linear(in_features=1568, out_features=512, bias=True)
        (1): ReLU()
        (2): Dropout(p=0.5, inplace=False)
        (3): Linear(in_features=512, out_features=10, bias=True)
      )
    )

    -----------------
    Begin train
    -----------------

    17:47:51 Epoch [00001/00010], Train Loss: 0.0349, Test Acc: 20662/21000 (98.4%)
    17:48:09 Epoch [00002/00010], Train Loss: 0.0288, Test Acc: 20744/21000 (98.8%)
    17:48:28 Epoch [00003/00010], Train Loss: 0.0108, Test Acc: 20676/21000 (98.5%)
    17:48:46 Epoch [00004/00010], Train Loss: 0.0098, Test Acc: 20753/21000 (98.8%)
    17:49:05 Epoch [00005/00010], Train Loss: 0.1214, Test Acc: 20784/21000 (99.0%)
    17:49:24 Epoch [00006/00010], Train Loss: 0.0177, Test Acc: 20760/21000 (98.9%)
    17:49:42 Epoch [00007/00010], Train Loss: 0.0281, Test Acc: 20802/21000 (99.1%)
    17:50:01 Epoch [00008/00010], Train Loss: 0.0005, Test Acc: 20825/21000 (99.2%)
    17:50:19 Epoch [00009/00010], Train Loss: 0.1858, Test Acc: 20759/21000 (98.9%)
    17:50:38 Epoch [00010/00010], Train Loss: 0.0006, Test Acc: 20810/21000 (99.1%)
    ```

## ライセンス

[MIT License](./LICENSE)

## GitHub

[LotFun](https://github.com/pystokes)
