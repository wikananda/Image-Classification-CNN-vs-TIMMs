# Image Classification: Classical CNN vs. TIMMs
Nyoman Wikananda Santana  
**Repository:** [wikananda/Image-Classification-CNN-vs-TIMMs](https://github.com/wikananda/Image-Classification-CNN-vs-TIMMs)  

This project gives basic implementation in image classification:  
- A custom or classical Convolutional Neural Network (CNN) architecture.  
- Pretrained models from the [TIMM](https://huggingface.co/timm) library (e.g., ResNet, SwiftFormer).  

The goal is to evaluate:  
- Accuration metrics (accuracy, precision, f1-score) 
- Resources efficiency (FLOPs, GMACs, #-parameters)

## 📁 Repository Structure  
```
/
├── dataset/               # Holds dataset files (images, labels, splits)
├── checkpoints/           # Saved trained models (checkpoints)
├── models/                # Model implementation
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
For my experiment, the model is trained using [AI vs Real image dataset](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images). I was only using the test set as my training, validation and test data due to my low computing power.
- Update `dataset.yaml` for train/val/test splits, image size, etc.  
- If you already processed your dataset (splitting and preprocessing), place it under `dataset/processed`.
- If not, place it under `dataset/raw` and run:
  ```bash
  DataModule.setup(processed=False)
  ```  

### Training  
- Adjust hyperparameters/model choice in `train.yaml`. For TIMM model, check out more [here](https://huggingface.co/timm/models).
- Check out `test_ai_vs_real.ipynb` for implementation example.
  
### Evaluation / Inference  
- After training, model checkpoints will be saved in `models/`.  
- Results (accuracy, confusion matrices, plots) will be stored under `results/`.

## 📝 Results Snapshot  
| Model                    | Accuracy  | Recall  | F1-Score  | GFLOPs  |  GMACs  | #-params  |
|--------------------------|-----------|---------|-----------|---------|---------|-----------|
| Custom CNN (baseline)    | 76.56 %   |  0.766  |   0.867   | 301.964 | 150.394 | 63.352 M  |
| SwiftFormer-xs (TIMM)    | 90.62 %   |  0.906  |   0.950   | 77.8282 | 38.5204 | 114.178 K |

## ✅ To Do  
- [ ] Train/Test CNN and ResNet as another baseline
- [ ] Check model latency

