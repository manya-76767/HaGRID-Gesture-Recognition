# Hand Gesture Recognition using HaGRID Dataset

## Overview

This project presents a comparative study of ResNet-50 and Swin Transformer for hand gesture recognition using a subset of the HaGRID dataset.

## Dataset

- Dataset: HaGRID
- Classes: 17
- Total Images: ~29,800
- Split Ratio: 70:15:15

### Classes

dislike, fist, four, like, mute, ok, one, palm, peace, peace_inverted, rock, stop, stop_inverted, three, three2, two_up, two_up_inverted

## Models

### ResNet-50
- Pretrained on ImageNet
- Input Size: 224x224

### Swin Transformer
- Transformer-based architecture
- Input Size: 224x224

## Results

## Results

| Model            | Best Validation Accuracy | Test Accuracy |
| ---------------- | -----------------------: | ------------: |
| ResNet-50        |                   92.77% |        93.72% |
| Swin Transformer |                   92.51% |        92.52% |

### ResNet-50 Metrics

* Accuracy: 93.72%
* Precision: 93.83%
* Recall: 93.72%
* F1 Score: 93.74%

### Swin Transformer Metrics

* Accuracy: 92.52%
* Precision: 92.64%
* Recall: 92.52%
* F1 Score: 92.51%


## Technologies

- Python
- PyTorch
- Torchvision
- timm
- NumPy
- Matplotlib

## Contributors

- Manya Arora
- Manya Pandey

## Repository Structure

- Code/ : Training and preprocessing scripts
- Figures/ : Accuracy, loss and confusion matrix plots
- Results/ : Evaluation metrics
- Paper/ : Screenshots and project documentation
