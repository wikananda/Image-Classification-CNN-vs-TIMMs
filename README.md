# Image Classification: Classical CNN vs. TIMMs
Nyoman Wikananda Santana  
**Repository:** [wikananda/Image-Classification-CNN-vs-TIMMs](https://github.com/wikananda/Image-Classification-CNN-vs-TIMMs)  

This project compares the performance of two paradigms in image classification:  
- A custom or classical Convolutional Neural Network (CNN) architecture.  
- Pretrained models from the [TIMM](https://huggingface.co/timm) library (e.g., ResNet, SwiftFormer).  

Currently, the main trained model is the SwiftFormer (via TIMM) variant; the classical CNN branch is planned/in progress.

The goal is to evaluate:  
- Accuration metrics (accuracy, precision, f1-score) 
- Resources efficiency (FLOPs, GMACs, #-parameters)

## 📁 Repository Structure  
```
/
├── dataset/               # Holds dataset files (images, labels, splits)
├── models/                # Saved trained models (checkpoints)
├── results/               # Logs, performance metrics, plots
├── test_imgs/             # Sample images for inference / demo
├── utils/                 # Utility scripts (data loading, transforms, etc.)
├── train.yaml             # Training configuration file (hyperparameters, dataset, model choice)
├── dataset.yaml           # Dataset configuration (path, classes, splits)
├── requirements.txt       # Python dependencies
└── test_ai_vs_real.ipynb  # Jupyter notebook demo
```

### Prerequisites  
- Install dependencies:  
  ```bash
  pip install -r requirements.txt
  ```
### Dataset Preparation  
- Update `dataset.yaml` for train/val/test splits, image size, etc.  
- If you already processed your dataset (splitting and preprocessing), place it under `dataset/processed`.
- If not, placed under `dataset/raw` and run DataModule.setup(processed=False).  

### Training  
- Adjust hyperparameters/model choice in `train.yaml`. For TIMM model, checkout more [here](https://huggingface.co/timm/models)
- Checkout `test_ai_vs_real.ipynb` for implementation example
  
### Evaluation / Inference  
- After training, model checkpoints will be saved in `models/`.  
- Results (accuracy, confusion matrices, plots) will be stored under `results/`.

## 📝 Results Snapshot  
| Model                    | Accuracy  | Recall  | F1-Score  | GFLOPs  |  GMACs  | #-params  |
|--------------------------|-----------|---------|-----------|---------|---------|-----------|
| Custom CNN (baseline)    | xx.xx %   |         |           |         |         |           |
| SwiftFormer (TIMM)       | 90.62 %   |  0.906  |   0.950   | 77.8282 | 38.5204 | 114.178 K |

## ✅ To Do  
- [ ] Train/Test CNN and ResNet as another baseline
- [ ] Check model latency

