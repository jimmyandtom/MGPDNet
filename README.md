# MGPD-Net: Active Multi-Granularity Prototypical Debiasing Network for Few-Shot Medical Image Segmentation

The Implementation of Paper: [MGPD-Net: Active Multi-Granularity Prototypical Debiasing Network for Few-Shot Medical Image Segmentation]()

![]()

#### Abstract

Labeled medical images for rare lesions are scarce in clinical practice, which limits the application of deep learning segmentation models that rely on large labeled datasets. Few-Shot Medical Image Segmentation (FSMIS) offers an effective solution, but it faces a critical challenge of feature entanglement, where class-consistent structural invariance (e.g., anatomical structure) and instance-specific appearance heterogeneity (e.g., texture details) are mixed together, leading to sampling bias and poor generalization. Existing methods use complex architectures or extra modalities, yet they cannot fully disentangle these mixed features. To solve this problem, we propose the Active Multi-Granularity Prototype Debiasing Network (MGPD-Net), which turns FSMIS from passive prototype matching into active query feature debiasing. Our method consists of the MultiGranularity Prototype Extraction (MPE) module for learnable semantic disentanglement (paradigm prototypes, appearance prototypes and orthogonal constraints) and the Asymmetric Dual-Stream Purification (ADSP) module to actively remove sampling bias. Extensive experiments on the SABS CT and CHAOS MRI datasets (1-shot setting) achieve MDSC scores of 76.27% and 80.68%. Without using auxiliary modalities or complex structures (e.g., Mamba), our method establishes an efficient disentanglement-calibration-enhancement framework, providing a lightweight and universal solution for the bias problem in FSMIS. The source code is available at
https://github.com/jimmyandtom/MGPDNet.

# Getting started

### Dependencies

Please install following essential dependencies:

```
dcm2nii
json5==0.8.5
jupyter==1.0.0
nibabel==2.5.1
numpy==1.22.0
opencv-python==4.5.5.62
Pillow>=8.1.1
sacred==0.8.2
scikit-image==0.18.3
SimpleITK==1.2.3
torch==1.10.2
torchvision=0.11.2
tqdm==4.62.3
```

### Data sets and pre-processing

Download:

1. [Combined Healthy Abdominal Organ Segmentation data set](https://chaos.grand-challenge.org/)
2. [Multi-Atlas Abdomen Labeling Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/218292)

Pre-processing is performed according to [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation/tree/2f2a22b74890cb9ad5e56ac234ea02b9f1c7a535) and we follow the procedure on their github repository.

### Training
Run `./script/train.sh`
### Acknowledgement

Our implementation is based on the works: [SSL-ALPNet](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation), [ADNet](https://github.com/sha168/ADNet), [QNet](https://github.com/ZJLAB-AMMI/Q-Net), [PAMI]([GitHub - YazhouZhu19/Partition-A-Medical-Image: [IEEE TIM 2024] Partition A Medical Image: Extracting Multiple Representative Sub-Regions for Few-shot Medical Image Segmentation](https://github.com/YazhouZhu19/Partition-A-Medical-Image))


