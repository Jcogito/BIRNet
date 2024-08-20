# [ICME 2024]BIRNet:Bi-directional Boundary-object Interaction and Refinement Network for Camouflaged Object Detection
> **Authors:** 
> Jicheng Yang,
> Qing Zhang,
> Yilin Zhao,
> Yuetong Li,
> Zeming Liu.

## 1. Overview
The overview of the proposed network BIRNet:
<img src="https://github.com/Jcogito/BIRNet/blob/main/results/overview.png">

## 2. Training/Testing

### 2.1. Prerequisites
1. Environmental Setups
    + Creating a virtual environment in terminal: `conda create -n birnet python=3.9`.
    
    + Installing necessary packages: `pip install -r requirements.txt`.
2. Data
    + The total dataset folder should like this:
	```
	-- DATA
	  | -- TrainDataset
	  |    | -- Imgs
	  |    | -- GT
 	  |    | -- Edge
	  | -- TestDataset
	  |    | -- CAMO
	  |    | -- CHAMELEON
 	  |    | -- COD10K
  	  |    | -- NC4K
 	```
3. Weights
    + Download the Res2Net or Swin-B weights, and then transfer them to the directory located at `./pre-weight/xx.pth`[download link (BaiduYun)](https://pan.baidu.com/s/1Nfe7nhMvz9giZb6NsxJ67Q?pwd=zrdd).
    + Download the pre-trained BIRNet weights, and then transfer them to the directory located at `./pre-weight/xx.pth`[download link (BaiduYun)](https://pan.baidu.com/s/19sGOYJFUQ5Si34k3gTs7SA?).

### 2.2. Results
  The pre-computed maps of BIRNet are available at [download link (BaiduYun)](https://pan.baidu.com/s/1_9Zm1ch5IJX0a2dPg5AB7A?pwd=37bj).
