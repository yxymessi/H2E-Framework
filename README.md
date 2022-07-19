# Identifying Hard Noise in Long-Tailed Sample Distribution
This project introduces a new challenge as Noisy Long-Tailed Classification (NLT) and three NLT benchmarks: ImageNet-NLT, Animal10-NLT, and Food101-NLT. The proposed H2E together with other baselines are also included. This project is the official implementation of the ECCV 2022(oral) paper Identifying Hard Noise in Long-Tailed Sample Distribution.


## Introduction
Conventional de-noising methods rely on the assumption that all samples are independent and identically distributed, so the resultant classifier, though disturbed by noise, can still easily identify the noises as the outliers of training distribution.However, the assumption is unrealistic in large-scale data that is inevitably long-tailed. Such imbalanced training data makes a classifier less discriminative for the tail classes, whose previously “easy” noises are now turned into “hard” ones—they are almost as outliers as the clean tail samples. We introduce this new challenge as Noisy Long-Tailed Classification (NLT). Not surprisingly, we find that most de-noising methods fail to identify the hard noises, resulting in significant performance drop on the three proposed NLT benchmarks: ImageNet-NLT, Animal10-NLT, and Food101-NLT.To this end, we design an iterative noisy learning framework called Hard-to-Easy (H2E). Our bootstrapping philosophy is to first learn a classifier as noise identifier invariant to the class and context distributional changes, reducing “hard” noises to “easy” ones, whose removal further improves the invariance. Experimental results show that our H2E out-performs state-of-the-art de-noising methods and their ablations on long-tailed settings while maintaining a stable performance on the conventional balanced settings.


## Citation
```bash
@inproceedings{yi2022identifying,
  title={Identifying Hard Noise in Long-Tailed Sample Distribution},
  author={Yi, Xuanyu and Tang, Kaihua and Hua, Xian-Sheng, and Lim,Joo-Hwee and Zhang, Hanwang},
  booktitle= {ECCV},
  year={2022}
}
```



<div align="center">
  <img src="./figs/figure1.png"/>
</div>


## Requirements

### Installation
Create a conda environment and install dependencies:

```bash
git clone https://github.com/yxymessi/H2E-Framework/tree/main/eccv_github
cd noise_longtail

conda create -n H2E python=3.8
conda activate H2E

pip install -r requirements.txt

```

### Benchmarks

We propose three benchmarks for the Noisy Long-Tailed (NLT) classification tasks: ImageNet-GLT and MSCOCO-GLT. Please follow the  links below to pre-prepare the datasets.
- For **ImageNet-NLT** [(link)](http://www.lujiang.info/cnlw.html), you can download the red&blue noisy mini-ImageNet.
- For **Animal10-NLT** [(link)](https://dm.kaist.ac.kr/datasets/animal-10n/), you can download Animal-10N.
- For **Food101-NLT** [(link)](https://kuanghuei.github.io/Food-101N/), you can download Food-101N.

Please refer to ./noise_longtail/construct_data for processing the dataset. We will publish a detailed benchmark geneartion instruction and release the annoatation for our exmperiments soon.


## H2E without Iteration

```bash 
bash ./noise_longtail/run_H2E.sh

```
## other baselines
we are still preparing and clear up the codebase zoo of long-tailed and denoise methods suitable for our NLT benchmarks.

## Acknowlegment
This repo benifits from [(link)](https://github.com/filipe-research/tutorial_noisylabels), thanks for their wonderful works.

