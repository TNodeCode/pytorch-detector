import torch
import torchvision.ops as ops
from data import extract_images_targets

def compute_iou(box1, box2):
    iou = ops.box_iou(box1.unsqueeze(0), box2.unsqueeze(0))
    return iou.item()

def compute_precision_recall(true_boxes, true_labels, pred_boxes, pred_labels, iou_threshold):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred_box, pred_label in zip(pred_boxes, pred_labels):
        iou_scores = [compute_iou(pred_box, true_box) for true_box, true_label in zip(true_boxes, true_labels) if true_label == pred_label]
        max_iou = max(iou_scores) if iou_scores else 0

        if max_iou >= iou_threshold:
            true_positives += 1
        else:
            false_positives += 1

    false_negatives = len(true_boxes) - true_positives

    if true_positives + false_positives != 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0
        
    if true_positives + false_negatives != 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0

    return precision, recall

def compute_ap(true_boxes, true_labels, pred_boxes, pred_labels, iou_threshold):
    precision, recall = compute_precision_recall(true_boxes, true_labels, pred_boxes, pred_labels, iou_threshold)
    ap = precision * recall
    return ap

def compute_map50(true_boxes, true_labels, pred_boxes, pred_labels):
    iou_threshold = 0.5
    unique_labels = torch.unique(torch.cat((true_labels, pred_labels)))

    ap_values = []
    for label in unique_labels:
        mask_true = true_labels == label
        mask_pred = pred_labels == label

        ap = compute_ap(true_boxes[mask_true], true_labels[mask_true],
                        pred_boxes[mask_pred], pred_labels[mask_pred], iou_threshold)
        ap_values.append(ap)

    mAP50 = torch.mean(torch.tensor(ap_values))
    return mAP50

def compute_map50_95(true_boxes, true_labels, pred_boxes, pred_labels):
    iou_thresholds = torch.arange(0.5, 1.0, 0.05)
    unique_labels = torch.unique(torch.cat((true_labels, pred_labels)))

    ap_values = []
    for label in unique_labels:
        mask_true = true_labels == label
        mask_pred = pred_labels == label

        ap = torch.mean(torch.tensor([compute_ap(true_boxes[mask_true], true_labels[mask_true],
                                                 pred_boxes[mask_pred], pred_labels[mask_pred], iou_thresh)
                                      for iou_thresh in iou_thresholds]))
        ap_values.append(ap)

    mAP50_95 = torch.mean(torch.tensor(ap_values))
    return mAP50_95

def compute_mAP_values(model, dataloader, n_batches_validation=10, device="cpu"):
    with torch.no_grad():
        mAP50_values = []
        mAP50_95_values = []
        for b, data in enumerate(dataloader):
            if b >= n_batches_validation:
                break
            images, targets = extract_images_targets(data, device=device)
            result = model(images)
            for i in range(len(result)):
                mAP50 = compute_map50(
                    true_boxes=targets[i]['boxes'],
                    true_labels=targets[i]['labels'],
                    pred_boxes=result[i]['boxes'],
                    pred_labels=result[i]['labels']
                )
                mAP50_95 = compute_map50_95(
                    true_boxes=targets[i]['boxes'],
                    true_labels=targets[i]['labels'],
                    pred_boxes=result[i]['boxes'],
                    pred_labels=result[i]['labels']
                )
                mAP50_values.append(mAP50)
                mAP50_95_values.append(mAP50_95)
        return torch.tensor(mAP50_values).mean(), torch.tensor(mAP50_95_values).mean()