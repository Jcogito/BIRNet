# [ICME 2024]BIRNet:Bi-directional Boundary-object Interaction and Refinement Network for Camouflaged Object Detection
> **Authors:** 
> Jicheng Yang,
> Qing Zhang,
> Yilin Zhao,
> Yuetong Li,
> Zeming Liu.

## 1. Overview
We introduce VSCode, a generalist model with novel 2D prompt learning, to jointly address four SOD tasks and three COD tasks. We utilize VST as the foundation model and introduce 2D prompts within the encoder-decoder architecture to learn domain and task-specific knowledge on two separate dimensions. A prompt discrimination loss helps disentangle peculiarities to benefit model optimization. VSCode outperforms state-of-the-art methods across six tasks on 26 datasets and exhibits zero-shot generalization to unseen tasks by combining 2D prompts, such as RGB-D COD.
<img src="https://github.com/Sssssuperior/VSCode/blob/main/method.png">

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
    + Download the Res2Net or Swin-B weights, and then transfer them to the directory located at `./pre-weight/xx.pth`[download link (Google Drive)](https://drive.google.com/).
    + Download the pre-trained BIRNet weights, and then transfer them to the directory located at `./pre-weight/xx.pth`[download link (Google Drive)](https://drive.google.com/).

### 2.2. Results
  The pre-computed maps of BIRNet are available at `./pre-weight/xx.pth`[download link (Google Drive)](https://drive.google.com/).

## 3. Citation
Please cite our paper if you find the work useful: 

	@inproceedings{BIRNet2024,
	title={Bi-directional Boundary-object interaction and refinement network for Camouflaged Object Detection},
	author={},
	booktitle={},
	pages = "",
	year={2024}
	}
