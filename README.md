# Deep Neural Network-based Numerical Material Test
-----
<a id="1"></a>
## Description
The Deep Neural Network-based Numerical Material Test (DNN-NMT) project provides neural network (NN) structure and datasets for estimating biaxial stress-strain curves of aluminum alloy sheets from a pole figure image of crystallographic texture. The NN provided in this project can be used on <a href="https://dl.sony.com/ja/app/">Neural Network Console</a> (NNC) developed by Sony Network Communications Inc.<br>

## Publications
1. K. Koenuma, A. Yamanaka, Ikumu Watanabe and Toshihiko Kuwabara, "Estimation of texture-dependent stress－strain curve and r-value of aluminum alloy sheet using deep learning", Journal of Japan Society for Technology of Plasticity, Vol. 61 No. 709 (2020), pp. 48-55. (in Japanese) <a href="https://doi.org/10.9773/sosei.61.48">doi.org/10.9773/sosei.61.48</a>
2. A. Yamanaka, R. Kamijyo, K. Koenuma, I. Watanabe and T. Kuwabara, "Deep neural network approach to estimate biaxial stress-strain curves of sheet metals", in preparation

## Contens
1. [Description](#1)
1. [Requirements](#2)
1. [Demonstration](#3)
1. [Usage](#4)
    1. [Importing our NN to NNC](#5)
    1. [Training NN](#6)
    1. [Re-training NN (Transfer learning)](#7)
    1. [Exporting trained NN](#8)
    1. [Estimation of biaxial stress-strain curves using trained NN](#9)
1. [Licence](#10)
1. [Developers](#11)

<a id="2"></a>
## Requirements
- Neural Network Console can be used on Windows 10. Please see: https://dl.sony.com/ja/app/
- Installation of Neural Network Console can be found: https://dl.sony.com/ja/app/
- Our trained NN has been tested using Python3.6.
- If you want to try [Demonstration](#3), you need to do the following procedures:

1. If you use use Anaconda (or Miniconda) environment, you need to do:
```bach
conda env create --file nnc_env.yaml
conda activate nnc_env
```

2.  If you do NOT use Anaconda (or Miniconda) environment, you need to install the following libraries:
```bash
pip install numpy matplotlib scipy
pip install nnabla
```

<a id="3"></a>
## Demonstration
- If you try this demonstration on your computer, please see [Requirements](#2).
- You can run our trained NN using the following command and estimate biaxial stress-strain curves.
```bash
python drawSScurve.py
```

<a id="4"></a>
## Usage
0. Overall usege of NNC please see: https://support.dl.sony.com/docs/

<a id="5"></a>
#### Importing our NN to NNC
- You can use the layer structure of our NN and the training parameters by importing "nntxt" file to NNC.

1. Launch NNC and select "New Project". <br>
![Fig1](./doc/fig1.png "Fig. 1")

2. Open "EDIT" tab and select "Import → nntxt, nnp, ONNX" by right-clicking a network graph.
![Fig. 2](./doc/fig2.png "Fig. 2")

3. Read "nntxt" file. <br>
ex) ./nnc_proj/model.nnp

4. If the name of imported NN is "MainRuntime", modify the name as "Main". <br>

- You can import the layer structure of NN and the training parameters used for the training by importing "nnp" file to NNC.


<a id="6"></a>
#### Training NN
- If you want to train our NN using the datasets provided in this project, you need to do the following procedures:

1. You can download the datasets from Yamanaka research group@TUAT.
<!-- TODO: ダウンロード先のリンクなどがあれば貼る -->
1. Save the datasets in the directory named "./trainingdata/".
1. Run "create_dataset.py" by using the following command, then you can create csv files which can be used in NNC.
- CSV file for training is saved in "./label/sscurve_train.csv".
- CSV file for validation is saved in "./label/sscurve_eval.csv".
```bash
python create_dataset.py
```

1. Open "DATASET" tab and select "Open Dataset". Then, read the above CSV files.
![Fig. 3](./doc/fig3.png "Fig. 3")
<!-- TODO: 図 -->

1. Open "EDIT" tab. Then, you need to connect the loss function, for example "SquerdError", to the Affine layers (Affine_4 and Affine_5).
1. Change from "y" to s" in the property (T.Dataset) of loss function for Affine_4. On the other hand, for Affine_5, change from "y" to "e".  
![Fig. 4](./doc/fig4.png "Fig. 4")
<!-- TODO: 図 -->

1. Open "CONFIG" tab and enter the values of Max Epoch and Batch Size. For example, you can use Max Epoch = 100 and Batch Size = 8.
1. Select the optimizing algorithm in "Optimizer" shown in "CONFIG" tab.
1. Start training of NN by pushing "F5 button".


<a id="7"></a>
#### Re-training NN (Transfer learning)
- If you want to re-train our NN using your datasets, please do the following procedures:

1. Open "EDIT" tab and select the parameters which you want to fix. For example, you can select convolution and BatchNormalization layers.
1. Enter 0 (zero) to "...LRateMultiplier" shown in the property of the selected laters. This prevents to change the parameters used in the trained NN.
![Fig. 5](./doc/fig5.png "Fig. 5")

1. Load your datasets from "DATASET" tab and start training.
1. <!-- TODO: 実験データのデータ構造に関する仕様を決めて，実験データから訓練データを作成するスクリプトをかく -->
1. Please see https://support.dl.sony.com/docs-ja/ for the training procedure.


<a id="8"></a>
#### Exporting trained NN
- You can export the trained NN by following procedures:

1. Open "TRAINING" tab and select "Export" by right-clicking the Results History. Then, you can select the format. <br>
ex) Export → NNP (Neural Network Libraries file format)
![Fig. 6](./doc/fig6.png "Fig. 6")

1. For the case shown in the above figure, the exported file is saved in the training result folder and is named as "model.nnp”.


#### Estimation of biaxial stress-strain curves using trained NN
- You can estimate biaxial stress-strain curves using the trained NN by following procedures:

1. Open "EDIT" tab and select the NN structure by right-clicking.
1. Select "Export → Python Code (NNabla)" and, then, Python code for the trained NN is copied to Clipboad. Note that you need to delete the loss function layers.
![Fig. 7](./doc/fig7.png "Fig. 7")

3. Create a python script named as "[project_name_of_NNC].py" in the directory "./nnc_proj" and paste the python code copied in the above Step 2.
4. Edit "estimate.py" provided in this project so that you can use the NN named "[project_name_of_NNC].py".
```python
# from nnc_proj.model import network  # ここを変える
from nnc_proj.project_name_of_NNC import network
nn.clear_parameters()
# nn.parameter.load_parameters('./nnc_proj/model.nnp')  # ここを変える
nn.parameter.load_parameters('NNCで作成したニューラルネットワークパラメータのファイル.nnp')
```

1. Run "drawSScurve.py" and then you obtain biaxial stress-strain curves.

## Licence

## Author
[Yamanaka Research Group @ TUAT](http://web.tuat.ac.jp/~yamanaka/)
