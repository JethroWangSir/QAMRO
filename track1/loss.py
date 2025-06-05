import torch
from scipy.stats import norm


def preference_aware_pamr_loss(preds, labels, preference_factor=7.0, margin_scale=0.2):
    preds = preds.squeeze()
    labels = labels.squeeze()
    B = preds.size(0)
    loss = 0.0
    count = 0
    
    if labels.max() != labels.min():
        norm_labels = (labels - labels.min()) / (labels.max() - labels.min())
    else:
        return torch.tensor(0.0, device=preds.device)
    
    for i in range(B):
        for j in range(i + 1, B):
            pred_i, pred_j = preds[i], preds[j]
            label_i, label_j = labels[i], labels[j]
            label_diff = label_i - label_j
            pred_diff = pred_i - pred_j
            
            if label_diff == 0:
                continue
                
            sign = 1 if label_diff > 0 else -1
            
            quality_level = max(norm_labels[i], norm_labels[j])
            weight = 1.0 + (preference_factor - 1.0) * quality_level
            
            margin = abs(label_diff) * margin_scale
            loss += weight * torch.relu(-sign * pred_diff + margin)
            count += 1
            
    if count == 0:
        return torch.tensor(0.0, device=preds.device)
    return loss / count
