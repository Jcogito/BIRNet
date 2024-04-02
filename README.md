# [ICME 2024]BIRNet:Bi-directional Boundary-object interaction and refinement network for Camouflaged Object Detection
> **Authors:** 
> Jicheng Yang,
> Qing Zhang,
> Yilin Zhao,
> Yuetong Li,
> Zeming Liu.

## 1. Training/Testing

### 1.1. Prerequisites
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
    + Download the Res2Net or Swin-B weights, and then transfer them to the directory located at `./pre-weight/xx.pth`[download link (Google Drive)](https://drive.google.com/).
    + Download the pre-trained BIRNet weights, and then transfer them to the directory located at `./pre-weight/xx.pth`[download link (Google Drive)](https://drive.google.com/).

### 1.2. Results
The pre-computed maps of BIRNet are available at `./pre-weight/xx.pth`[download link (Google Drive)](https://drive.google.com/).

## 2. Citation
Please cite our paper if you find the work useful: 

	@inproceedings{,
	title={},
	author={},
	booktitle={},
	pages = "",
	year={2024}
	}
