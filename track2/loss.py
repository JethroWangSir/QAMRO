import torch
from scipy.stats import norm
    

def preference_aware_pamr_loss(preds, labels, task='pq', preference_factor=7.0, margin_scale=0.2):
    preds = preds.squeeze()
    labels = labels.squeeze()
    B = preds.size(0)
    loss = 0.0
    count = 0

    # Define weights based on task and pair counts (descending order)
    if task == 'pq':
        weights_map = {
            (6.0, 7.0): 9.5,
            (7.0, 8.0): 8.5,
            (5.0, 6.0): 7.5,
            (4.0, 5.0): 6.5,
            (8.0, 9.0): 5.5,
            (3.0, 4.0): 4.5,
            (1.0, 2.0): 0.0,
            (2.0, 3.0): 0.0,
            (9.0, 10.0): 0.0,
        }
    elif task == 'pc':
        weights_map = {
            (3.0, 4.0): 9.5, # Example weights based on PC sorted order
            (1.0, 2.0): 8.5,
            (4.0, 5.0): 7.5,
            (2.0, 3.0): 6.5,
            (6.0, 7.0): 5.5,
            (5.0, 6.0): 4.5,
            (7.0, 8.0): 3.5,
            (8.0, 9.0): 0.0,
            (9.0, 10.0): 0.0,
        }
    elif task == 'ce':
        weights_map = {
            (6.0, 7.0): 9.5, # Example weights based on CE sorted order
            (5.0, 6.0): 8.5,
            (3.0, 4.0): 7.5,
            (4.0, 5.0): 6.5,
            (7.0, 8.0): 5.5,
            (2.0, 3.0): 4.5,
            (1.0, 2.0): 0.0,
            (8.0, 9.0): 0.0,
            (9.0, 10.0): 0.0,
        }
    elif task == 'cu':
        weights_map = {
            (6.0, 7.0): 9.5, # Example weights based on CU sorted order
            (7.0, 8.0): 8.5,
            (5.0, 6.0): 7.5,
            (4.0, 5.0): 6.5,
            (3.0, 4.0): 5.5,
            (8.0, 9.0): 4.5,
            (1.0, 2.0): 0.0,
            (2.0, 3.0): 0.0,
            (9.0, 10.0): 0.0,
        }
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'pq', 'pc', 'ce', or 'cu'.")

    # Normalize weights to 0~1
    weights_values = torch.tensor(list(weights_map.values()))
    if weights_values.max() > weights_values.min():
        norm_weights_map = {k: (v - weights_values.min()) / (weights_values.max() - weights_values.min())
                            for k, v in weights_map.items()}
    else:
        norm_weights_map = weights_map # If all weights are the same, no need to normalize

    for i in range(B):
        for j in range(i + 1, B):
            pred_i, pred_j = preds[i], preds[j]
            label_i, label_j = labels[i].item(), labels[j].item() # Get scalar values for comparison
            label_diff = label_i - label_j
            pred_diff = pred_i - pred_j

            if label_diff == 0:
                continue

            sign = 1 if label_diff > 0 else -1

            # Determine the MOS label interval for each label
            def get_interval(label):
                if 1.0 <= label < 2.0: return (1.0, 2.0)
                elif 2.0 <= label < 3.0: return (2.0, 3.0)
                elif 3.0 <= label < 4.0: return (3.0, 4.0)
                elif 4.0 <= label < 5.0: return (4.0, 5.0)
                elif 5.0 <= label < 6.0: return (5.0, 6.0)
                elif 6.0 <= label < 7.0: return (6.0, 7.0)
                elif 7.0 <= label < 8.0: return (7.0, 8.0)
                elif 8.0 <= label < 9.0: return (8.0, 9.0)
                elif 9.0 <= label <= 10.0: return (9.0, 10.0)
                return None

            interval_i = get_interval(label_i)
            interval_j = get_interval(label_j)

            if interval_i is None or interval_j is None:
                continue # Skip if label is out of defined intervals

            # Get the normalized weight based on the intervals
            weight_i = norm_weights_map.get(interval_i, 0.0)
            weight_j = norm_weights_map.get(interval_j, 0.0)
            quality_level = max(weight_i, weight_j) # Using the normalized weights

            final_weight = 1.0 + (preference_factor - 1.0) * quality_level

            margin = abs(label_diff) * margin_scale
            loss += final_weight * torch.relu(-sign * pred_diff + margin)
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=preds.device)
    return loss / count
