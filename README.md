<h1 align="center"> Spectral index-driven FCN model automatic training for water extraction from multispectral imagery </h1>

<h5 align="center"><em>Zhenshi Li, Xueliang Zhang, and Pengfeng Xiao</em></h5>

## Introduction

This is the official repository for the paper [“Spectral index-driven FCN model training for water extraction from multispectral imagery”](https://www.sciencedirect.com/science/article/abs/pii/S0924271622002283).

**Abstract:** Terrestrial water is a fundamental component of the land surface. Accurate and robust water delineation of surface water is rather challenging due to the high intra-class variability and the spectral similarity with shadows and other dark surfaces. This study proposes a new method called the water index-driven deep fully convolutional neural network (WIDFCN) for high-accuracy water delineation with no need to collect samples manually. We formulate water delineation as a semantic labeling problem and solve it by training a deep fully convolutional network (FCN), whose capability of effectively extracting multilevel spatial and spectral features is exploited for discriminating water from complex surroundings. The main obstacle of using FCN, which requires a large volume of labeled training samples, is settled by utilizing the water recognition ability of the water spectral index (WI). Specifically, the training samples are automatically generated from WI by extracting a high-precision but incomplete water mask at first, which is then expanded to enhance the completeness. This strategy ensures the high quality of the automatically generated training samples and thus the water extraction performance of the trained FCN model. Twelve test sites from Sentinel-2 imagery with various water delineation challenges all over the world are used to assess the performance relative to that of supervised and unsupervised classification methods and water spectral index thresholding methods, in terms of Kappa coefficient, precision, recall, and F1-score. Overall, WIDFCN can achieve the highest precision and comparable recall, leading to the best water delineation accuracy with Kappa coefficient 0.9673 and F1-score 0.9696, as well as the lowest fluctuation in terms of various test sites. The results further demonstrate that WIDFCN can effectively deal with the scale and spectra variance of surface water, and has distinct robustness with respect to different kinds of shadows, including building, mountain, and cloud shadows. The findings in this study demonstrate a novel, robust, low-cost, and manual labor-free water delineation method that performs well in terms of both precision and completeness. Moreover, the core idea could provide a reference for extraction of geographic information by utilizing FCN models with no needs of manually labeling costs.

## Usage

Employ *infer_water* with your weight path, data root and output root.

## Trained model

The trained model can be downloaded from [here](https://pan.baidu.com/s/12k8OAa4KFBzlcVfZBbrPdQ) (Code:o006)

## Test dataset

The test dataset can be downloaded from [here](https://pan.baidu.com/s/1VuGhywVtVOKb8EwxHDJg4Q) (Code:j57i)

## Visual results

<img src=Figure/FIG1.png width="90%">

<img src=Figure/FIG2.png width="90%">

<img src=Figure/FIG3.png width="90%">

<img src=Figure/FIG4.png width="90%">

<img src=Figure/FIG5.png width="90%">

<img src=Figure/FIG6.png width="90%">

</div>

## Acknowledgement

- Many thanks to [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception).

## Statement

- Please cite our paper if our work is useful to your research.

  ```

    @article{li2022spectral,

    title={Spectral index-driven FCN model training for water extraction from multispectral imagery},

    author={Li, Zhenshi and Zhang, Xueliang and Xiao, Pengfeng},

    journal={ISPRS Journal of Photogrammetry and Remote Sensing},

    volume={192},

    pages={344--360},

    year={2022},

    publisher={Elsevier}

    }

  ```

- Any questions please contact Lzhenshi@outlook.com.
