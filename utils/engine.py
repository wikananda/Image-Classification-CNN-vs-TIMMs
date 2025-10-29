import numpy as np
from PIL import Image
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
from torch import nn, optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from utils.datamodule import _build_transforms
from models.cnn import CNN
from models.timm_backbone import TimmClassifier
from utils.metric_utils import metrics, _count_parameter


with open("dataset.yaml", 'r') as f:
    dataset_cfg = yaml.safe_load(f)

with open("train.yaml", 'r') as f:
    train_cfg = yaml.safe_load(f)
model_cfg = train_cfg.get("model_cfg", {})

device = train_cfg.get('device', 'cpu') # default using cpu
_test_transform = _build_transforms(dataset_cfg)["test"]

def _save_train_plot(curves, show=False):
    acc = curves['train_accs']
    val_acc = curves['val_accs']
    loss = curves['train_losses']
    val_loss = curves['val_losses']
    epochs = len(loss)
    model_name = _get_model_name()
    epoch_idx = np.arange(1, epochs + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_idx, acc, label='Training Accuracy')
    plt.plot(epoch_idx, val_acc, label='Validation Accuracy')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epoch_idx, loss, label='Training Loss')
    plt.plot(epoch_idx, val_loss, label='Validation Loss')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f"{train_cfg['root_results']}/{train_cfg['project_name']}_{model_name}.png")
    if show:
        plt.show()

def _save_metrics_txt(metrics, path):
    with open(path, "w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

def _get_model_name():
    return train_cfg['model_name'] if train_cfg['model_name'] == 'cnn' else f"{train_cfg.get('model_cfg').get('timm')['name']}"


def _build_model():
    model_name = train_cfg["model_name"]
    if model_name == "cnn":
        cfg = model_cfg.get("cnn", model_cfg)
        cfg.setdefault("input_size", dataset_cfg.get("img_size", 128))
        cfg.setdefault("in_channels", train_cfg.get("in_channels", 3))
        model = CNN(num_classes=train_cfg["num_classes"], **cfg)
    elif model_name == "timm":
        cfg = model_cfg.get("timm", model_cfg)
        cfg = dict(cfg)
        backbone_name = cfg['name']
        pretrained = cfg.get('pretrained', True)
        freeze_backbone = cfg.get('freeze_backbone', True)
        kwargs = cfg.get('create_kwargs', {})
        model = TimmClassifier(
            name=backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            cache_dir=train_cfg["root_checkpoints"],
            **kwargs,
        )
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    
    return model.to(device)

def _select_criterion():
    criterion_name = train_cfg['criterion']
    if criterion_name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Criterion {criterion_name} not recognized.")
    
def _select_optimizer(model_parameters, lr, weight_decay):
    optimizer_name = train_cfg['optimizer']
    if optimizer_name == 'Adam':
        return optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        return optim.SGD(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        return optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not recognized.")

def train(dm):
    print(f"Using device: {device}")

    train_loader = dm.train_loader()
    val_loader = dm.val_loader()
    model = _build_model()
    params = _count_parameter(model)
    criterion = _select_criterion()
    optimizer = _select_optimizer(model.parameters(),
                           lr=float(train_cfg["lr"]),
                           weight_decay=float(train_cfg["weight_decay"]))
    # print("model: ", train_cfg[''])
    # model_name = train_cfg['model_name'] if train_cfg['model_name'] == 'cnn' else f"{train_cfg.get('model_cfg').get('timm')['name']}"
    model_name = _get_model_name()
    print(f"Using model {model_name} with params: {params}")
    num_epochs = train_cfg["num_epochs"]

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        train_loss = 0.0
        val_loss = 0.0
        train_preds = []
        train_labels = []
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) # predict
            loss = criterion(outputs, labels) # compute loss
            optimizer.zero_grad() # reset gradients
            loss.backward() # backpropagation
            optimizer.step() # update weights
            train_loss += loss.item()
            # for calculating metrics
            pred = outputs.argmax(dim=1)
            train_preds.extend(pred.detach().cpu().tolist())
            train_labels.extend(labels.detach().cpu().tolist())
        train_acc = accuracy_score(train_labels, train_preds)
        train_accs.append(train_acc)
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # validation phase
        val_preds = []
        val_labels = []
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                # for calculating metrics
                pred = outputs.argmax(dim=1)
                val_preds.extend(pred.detach().cpu().tolist())
                val_labels.extend(labels.detach().cpu().tolist())
            val_acc = accuracy_score(val_labels, val_preds)
            val_accs.append(val_acc)
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Accuracy: {train_acc:.4f} | Val Accuracy: {val_acc:.4f}")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, f"{train_cfg['root_checkpoints']}/{train_cfg['project_name']}_{model_name}.pt")
    curves = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs
    }
    np.savez(
        f"{train_cfg['root_results']}/{train_cfg['project_name']}_{model_name}_curves.npz",
        train_losses=np.asarray(train_losses, dtype=np.float32),
        val_losses=np.asarray(val_losses, dtype=np.float32),
        train_accs=np.asarray(train_accs, dtype=np.float32),
        val_accs=np.asarray(val_accs, dtype=np.float32),
    )
    _save_train_plot(curves)
    return model, curves

def test(dm):
    # testing phase
    test_loader = dm.test_loader()
    model = _build_model()
    model_name = _get_model_name()
    checkpoint = torch.load(f"{train_cfg['root_checkpoints']}/{train_cfg['project_name']}_{model_name}.pt")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    input_shape = (dataset_cfg['batch_size'], 3, dataset_cfg['img_size'], dataset_cfg['img_size'])
    print("input shape: ", input_shape)
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) # output highest, index
        test_preds = predicted.detach().cpu().tolist()
        test_targets = labels.detach().cpu().tolist()
    metric, cm = metrics(test_preds, test_targets, model, input_shape)
    _save_metrics_txt(metric, f"{train_cfg['root_results']}/{model_name}.txt")
    print("\nMETRICS =================================")
    print(f"Accuracy: {metric['accuracy']*100:.2f}%")
    print(f"Precision: {metric['precision']:.4f}")
    print(f"Recall: {metric['recall']:.4f}")
    print(f"F1-Score: {metric['f1_score']:.4f}")
    print("\nEFFICIENCY ==============================")
    print(f"FLOPs: {metric['flops']}")
    print(f"MACs: {metric['macs']}" )
    print(f"Params: {metric['params']}")
    return metric, cm

def predict_single_img(image_path):
    model = _build_model()
    model_name = _get_model_name()
    checkpoint = torch.load(f"{train_cfg['root_checkpoints']}/{train_cfg['project_name']}_{model_name}.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    image = Image.open(image_path).convert("RGB")
    image = _test_transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return predicted.item(), confidence.item(), image
