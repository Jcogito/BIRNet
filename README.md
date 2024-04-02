# [ICME 2024]BIRNet:Bi-directional Boundary-object interaction and refinement network for Camouflaged Object Detection
> **Authors:** 
> Jicheng Yang,
> Qing Zhang,
> Yilin Zhao,
> Yuetong Li,
> Zeming Liu.

## 1.Training/Testing

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
+ downloading Res2Net or Swin-B weights and move it into `./pre-weight/res2net50_v1b_26w_4s-3cf99910.pth`[download link (Google Drive)](https://drive.google.com/).
+ downloading pretrained weights`Res2Net or Swin-B` and move it into `./pre-weight/BIR_res2net.pth`[download link (Google Drive)](https://drive.google.com/).

## 3. Citation

Please cite our paper if you find the work useful: 

	@inproceedings{,
	title={},
	author={},
	booktitle={},
	pages = "",
	year={2024}
	}
