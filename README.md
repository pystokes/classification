# Classification

画像分類用のサンプルコード。確認用としてMNISTの手書き文字分類問題も実装済み。

## 環境情報

下記環境にて動作確認を実施。

- Windows 10
- Python 3.8.2
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
git clone https://github.com/pystokes/classification.git
```

クローンしたリポジトリに移動し，必要なライブラリをインストールして完了。

```bash
cd classification
pip install -r requirements.txt
```

#### [環境情報](#環境情報)を満たしていない場合

1. PyTorchのインストール

    [公式サイト](https://pytorch.org/get-started/locally/)から環境にあった方法でインストールする。

    例：[環境情報](#環境情報)を満たしている場合は以下のようになる。

    ```bash
    pip install torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
    ```

2. scikit-learnのインストール

    ```bash
    pip install sklearn
    ```

3. このリポジトリを任意のディレクトリにクローンして完了

    ```bash
    git clone https://github.com/pystokes/classification.git
    ```

## 使い方

### データの準備

#### MNISTの場合

MNISTで動作確認を行いたい場合は，事前に以下の手順にてデータセットを用意する。

```bash
cd tools
python prepare_mnist.py
cd ..
```

#### オリジナルデータを用いる場合

データを下記の形式で準備する(分類数がclass1～3の計3クラスの場合)

```bash
data_home
    ├─test
    │  ├─class1
    │  │  ├─class1-1.jpg
    │  │  ├─class1-2.jpg
    │  │  └─...
    │  ├─class2
    │  │  ├─class2-1.jpg
    │  │  ├─class2-2.jpg
    │  │  └─...
    │  └─class3
    │     ├─class3-1.jpg
    │     ├─class3-2.jpg
    │     └─...
    └─train
        ├─class1
        │  ├─class1-1.jpg
        │  ├─class1-2.jpg
        │  └─...
        ├─class2
        │  ├─class2-1.jpg
        │  ├─class2-2.jpg
        │  └─...
        └─class3
            ├─class3-1.jpg
            ├─class3-2.jpg
            └─...
```

1. [config.json](./config.json)の各種パラメータを設定する

    ```bash
    {
        # 学習時のパラメータ
        "train": {
            # 上記でデータセットを格納したディレクトリ（MNISTの場合，デフォルトは"./mnist_data"）
            "data_home": "./mnist_data",
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

2. 学習を実行

    ```bash
    python main.py
    ```

    上記のコマンドを実行すると以下のようなログが標準出力に表示される。

    ```bash
    $ python main.py

    ADD STDOUT LATER
    ```

## License

[MIT License](./LICENSE)

## GitHub

[pystokes](https://github.com/pystokes)
