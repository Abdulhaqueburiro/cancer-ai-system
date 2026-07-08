# Model 2 - Cancer Type Classification

## Overview
5-class classification model to identify cancer type from histopathology images.

## Classes
| Label | Class | Type |
|-------|-------|------|
| 0 | Lung Adenocarcinoma | Cancer |
| 1 | Lung Squamous Cell Carcinoma | Cancer |
| 2 | Colon Adenocarcinoma | Cancer |
| 3 | Lung Normal | Normal |
| 4 | Colon Normal | Normal |

## Performance
| Metric | Value |
|--------|-------|
| Val Accuracy | 94.19% |
| Test Accuracy | 94.0% |
| Val AUC | 0.9942 |
| Top-2 Accuracy | 99%+ |

## Dataset
- LC25000 Histopathology Dataset
- 25,000 images (224x224px)
- Perfectly balanced (5000 per class)

## Architecture
- Base: ConvNeXt-Base (ImageNet pretrained)
- Head: GAP -> BN -> Dense(512) -> Dropout(0.5) -> Dense(256) -> Softmax(5)
- Phase 1: Frozen base (LR=1e-4, 15 epochs)
- Phase 2: Fine-tuning (LR=1e-5, 10 epochs)

## EDA

### Class Distribution
![](results/eda_class_distribution.png)

### Sample Images
![](results/eda_sample_images.png)

### Pixel Intensity
![](results/eda_pixel_intensity.png)

### RGB Channel Analysis
![](results/eda_rgb_channels.png)

### Spatial Analysis
![](results/eda_spatial.png)

### t-SNE Feature Space
![](results/eda_tsne.png)

## Evaluation

### Confusion Matrix
![](results/evaluation_results.png)

### Predictions
![](results/predictions_model2.png)

### Grad-CAM
![](results/gradcam_model2.png)

## Links
- Model: https://huggingface.co/Abdulhaque/cancer-type-classification
- Demo: https://huggingface.co/spaces/Abdulhaque/cancer-type-classification-app
