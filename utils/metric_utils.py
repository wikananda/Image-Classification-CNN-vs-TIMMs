import numpy as np
from calflops import calculate_flops
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

def _count_parameter(model):
    return sum(p.numel() for p in model.parameters())

def metrics(predictions, targets, model, input_shape):
    """
    Compute classification metrics: precision, recall, F1-score, accuracy
    """
    preds = np.asarray(predictions, dtype=np.int64)
    targs = np.asarray(targets, dtype=np.int64)

    accuracy = accuracy_score(targs, preds)
    precision = precision_score(targs, preds, average='weighted', zero_division=0)
    recall = recall_score(targs, preds, average='weighted', zero_division=0)
    f1 = f1_score(targs, preds, average='weighted', zero_division=0)
    cm = confusion_matrix(targs, preds)

    flops, macs, params = profile_model(model, input_shape, True)

    metrics = {
        'params': params,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'flops': flops,
        'macs': macs
    }
    return metrics, cm

def profile_model(model, input_shape, output=False):
    # count parameters
    # params = _count_parameter(model)
    flops, macs, params = calculate_flops(
        model=model,
        input_shape=input_shape,
        output_as_string=output,
        output_precision=4
    )
    return flops, macs, params
    
