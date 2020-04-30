# DNN-NMT
-----

## Description (概要)
DNN-NMT (Deep Neural Network-based Numerical Material Test) project provides a neural network (NN) structure for estimating biaxial stress-strain curves of aluminum alloy sheet from a pole figure image of crystallographic texture. DNN-NMT is based on <a href="https://dl.sony.com/ja/app/">Neural Network Console</a> (NNC) developed by Sony Network Communications Inc.<br>

DNN-NMT (Deep Neural Network-based Numerical Material Test) プロジェクトでは, 集合組織を表す極点図の画像情報からアルミニウム合金板材の二軸応力-ひずみ曲線を推定するためのニューラルネットワーク(NN)の構造を提供します. DNN-NMTは, Sony Network Communicationsが開発した<a href="https://dl.sony.com/ja/app/">Neural Network Console</a> (NNC)に基づいています. <br>

## Publication (出版物)
1. K. Koenuma, A. Yamanaka, Ikumu Watanabe and Toshihiko Kuwabara, "Estimation of texture-dependent stress－strain curve and r-value of aluminum alloy sheet using deep learning", Journal of Japan Society for Technology of Plasticity, Vol. 61 No. 709 (2020), pp. 48-55. (in Japanese) <a href="https://doi.org/10.9773/sosei.61.48">doi.org/10.9773/sosei.61.48</a>
2. A. Yamanaka, R. Kamijyo, K. Koenuma, I. Watanabe and T. Kuwabara, "Deep neural network approach to estimate biaxial stress-strain curves of sheet metals", in preparation

## Contens (目次)
1. [Description(概要)](#description)
1. [Requirements (環境構築)](#requirements)
1. [Demonstration (訓練済みDNN-NMTの使い方)](#demo)
1. [Usage](#usage)
    1. [Importing a layer structure of DNN-MNT to NNC (DNN-MNTのNNCへのインポート)](#DNNNMT構造のインポート)
    1. [Training DNN-NMT (DNN-NMTの訓練)](#DNN-NMTの訓練)
    1. [Importing a layer structure of trained DNN-MNT to NNC (訓練済みDNN-NMTのNNCへのインポート)](#訓練済みモデルのインポート)
    1. [Re-training DNN-NMT (DNN-NMTの再訓練（転移学習）](#ニューラルネットワークの転移学習)
    1. [Exporting trained DNN-NMT (訓練済みDNN-NMTのエクスポート)](#訓練済みモデルのエクスポート)
    1. [Estimation of stress-strain curve using trained DNN-NMT (訓練済みDNN-NMTを用いた応力-ひずみ曲線の推定)](#訓練済みモデルを用いた応力ーひずみ曲線の推定)
1. [Licence (ライセンス)](#licence)
1. [Developers (開発者)](#author)


## Requirements
- Neural Network Console can be used on Windows 10. <br>
Neural Network Consoleは, Windows OS上で動作します.

- Installation of Neural Network Console can be found [here](https://dl.sony.com).<br>
  Neural Network Consoleのインストール方法は下記サイト参照してください.   
[Windows版 - Neural Network Console - Sony](https://dl.sony.com/ja/app/)

- Trained DNN-NMT has been tested on Python3.6. <br>
訓練済みのDNN-NMTは, Python3.6で確認済みです.

- If you want to try [Demo](#demo), you need: <br>
 下記の[Demo](#demo)を試される場合は, 以下の環境構築してください.
1. If you use use Anaconda (Miniconda), you need to do: <br>
もし, Anaconda (Miniconda)を使用されている場合は, 以下のようにNNCを使える環境が必要です.

```bach
conda env create --file nnc_env.yaml
conda activate nnc_env
```

2.  If you do NOT use Anaconda (Miniconda), you can install libraries: <br>
もしくは下記により必要なパッケージをインストールしてください.

```bash
pip install numpy matplotlib scipy
pip install nnabla
```


## Demo
- If you try this Demo, please see [Requirements](#requirements).<br>
このDemoを試される場合には, [Requirements](#requirements)をご覧ください.

- You can run the trained DNN-NMT with the following command and estimate biaxial stress-strain curves. <br>
下記コマンドによりdrawSScurve.pyを実行することで, 訓練済みDNN-NMTを用いて二軸応力-ひずみ曲線を推定することができます.

```bash
python drawSScurve.py
```


## Usage
- [Official documents for Neural Network Console](https://support.dl.sony.com/docs/)<br>
- [Neural Network Consoleの公式ドキュメント](https://support.dl.sony.com/docs-ja/)

#### DNN-NMTのNNCへのインポート
- 以下の方法で, nntxtファイルをNNCにインポートすることで, DNN-NMTの構造や学習時の設定を読み込むことができます.  

1. NNCを起動し, New Projectを選択する.

![Fig1](./doc/fig1.png "Fig. 1")

2. EDITタブを開き, ネットワークグラフ上を右クリックして表示されるポップアップメニューから  
Import → nntxt, nnp, ONNXを選択する.

![Fig. 2](./doc/fig2.png "Fig. 2")

3. nntxtファイルを読み込む.  
ex) ./nnc_proj/model.nnp

4. 読み込んだニューラルネットワークの名前がMainRuntimeになっている場合は, Mainに書き換えておく.

- 上記3において, nnpファイルをNNCにインポートすることで, DNN-NMTの構造に加えて訓練済みのパラメータなどを読み込むことができます.

#### DNN-NMTの訓練
- 本プロジェクトで提供する訓練データを用いて, DNN-NMTを訓練するには, 以下の手順にしたがってください.

<!-- TODO: ダウンロード先のリンクなどがあれば貼る -->
1. ダウンロードした訓練データを./trainingdata/ フォルダに入れておく.

2. 下記コマンドによりcreate_dataset.pyを実行することで, NNCで使用可能なデータセットのcsvファイルを作成することができます.  
訓練データのcsvファイル： ./label/sscurve_train.csv  
検証データのcsvファイル： ./label/sscurve_eval.csv  

```bash
python create_dataset.py
```

3. NNCのDATASETタブを開き, Open Datasetを選択します(または, ショートカットCtrl + o). エクスプローラを操作し, 上記の手順で作成したcsvファイルを読み込みます.

![Fig. 3](./doc/fig3.png "Fig. 3")
<!-- TODO: 図 -->

4. EDITタブを開き, ニューラルネットワークに誤差関数として二乗誤差(SquerdError)を接続する. SquerdError以外の回帰問題用の誤差関数(例えば, HuberLoss)であってもよい.  
DNN-NMTには二つの出力層(Affine_4, Affine_5)があるため, 二つの出力層どちらに対しても誤差関数を接続する.

5. Affine_4に接続した誤差関数のプロパティからT.Datasetを選択し, y → sと書き換える. もう一方の誤差関数では, y → eと書き換える.

![Fig. 4](./doc/fig4.png "Fig. 4")
<!-- TODO: 図 -->

6. CONFIGタブを開き, Global Configの設定画面からMax EpochやBatch Sizeを設定する. ex) Max Epoch: 100, Batch Size: 8  
ただし, Batch Sizeを訓練データセットの数より大きくすることができないので注意する.  
訓練時に使用される最適化アルゴリズムはCONFIGタブのOptimizerから選択する.

7. 訓練実行ボタンをクリック(あるいは, ショートカット F5)することで, ニューラルネットワークの訓練が開始される.

#### ニューラルネットワークの訓練(転移学習)
- 訓練済みのDNN-NMTを新たなデータセットで再訓練する場合には, 以下の手順に従ってください.

1. EDITタブ上のニューラルネットワークのうち, パラメータを固定するレイヤーを選択する.  
例えば, 最後のAffineレイヤーを除くレイヤー (ConvolutionやBatchNormalization) を選択する.

2. 選択したレイヤープロパティから, LRateMultiplierを0に設定する.  
この操作により, 訓練済みのニューラルネットワークのパラメータが  
新たに訓練するデータで更新されないようにする.

![Fig. 5](./doc/fig5.png "Fig. 5")

3. DATASETタブで転移先のデータを読み込み, 訓練を実行する.  
LRateMultiplierプロパティが0に設定されていないパラメータのみ更新される.

4. <!-- TODO: 実験データのデータ構造に関する仕様を決めて，実験データから訓練データを作成するスクリプトをかく -->

5. 訓練の実施方法は, [公式ドキュメント](https://support.dl.sony.com/docs-ja/)を参照してください.

#### 訓練済みモデルのエクスポート
- 訓練済みのDNN-NMTをエクスポートする場合には, 下記の手順に従ってください.

1. NNCのTRAININGタブの左側に表示される学習結果リストから, エクスポートする学習結果を選択する.

2. 選択した学習結果を右クリックして表示されるポップアップメニューからExportを選択し, その後Exportするファイル形式を選択する.  
ex) Export → NNP (Neural Network Libraries file format)

![Fig. 6](./doc/fig6.png "Fig. 6")

3. 上の例の場合, Exportされたファイルは, 学習結果ファイルの格納されたフォルダに  
”model.nnp”のファイル名で生成される.

 - 学習結果ファイルは, プロジェクトファイル名の拡張子を除いたものに,  
  “.files”を付加した名前のフォルダの下,  
  さらに学習を実行した日付・時刻の名前のフォルダの下に保存されている.

#### 訓練済みモデルを用いた応力ーひずみ曲線の推定
- 訓練済みのDNN-NMTを用いて, 応力ーひずみ曲線の推定を行う方法は, 以下の通りです.

1. NNCのEDITタブを開き, ネットワークグラフ上の作成したニューラルネットワークを右クリックで選択する.

2. ポップアップメニューからExport → Python Code (NNabla)を選択する.  クリップボードにニューラルネットワークのpythonコードがコピーされる. このとき, 誤差関数のレイヤーは削除しておく.

![Fig. 7](./doc/fig7.png "Fig. 7")

3. [project_name_of_NNC].pyのようなpythonスクリプトを./nnc_projの下に作成し, 2. でコピーされたpythonコードを貼り付ける.

4. estimate.pyを編集し, 上の手順で作成したニューラルネットワークを使用するよう書き換える.

```python
# from nnc_proj.model import network  # ここを変える
from nnc_proj.project_name_of_NNC import network

nn.clear_parameters()
# nn.parameter.load_parameters('./nnc_proj/model.nnp')  # ここを変える
nn.parameter.load_parameters('NNCで作成したニューラルネットワークパラメータのファイル.nnp')
```

1. drawSScurve.pyを実行する. 詳細は, [Demo](#demo)を参照されたい.

## Licence

## Author
[Yamanaka Research Group @ TUAT](http://web.tuat.ac.jp/~yamanaka/)
