import torch
from torchvision.datasets import CocoDetection


def build_coco_dataset(root: str, annFile: str, transform):
    return CocoDetection(root=root, annFile=annFile, transform=transform)

def build_dataloader(dataset, batch_size: int, collate_fn=None):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

def collate_fn_coco(data):
    return data

def extract_images_targets(data, device="cpu"):    
    images = []
    targets = []
    for image, annotations in data:
        images.append(image.to(device))
        bboxes = list(map(lambda x: [x['bbox'][0], x['bbox'][1], x['bbox'][0]+x['bbox'][2], x['bbox'][1]+x['bbox'][3]], annotations))
        labels = list(map(lambda x: int(x['category_id'])+1, annotations))
        target = {}
        target["boxes"] = torch.tensor(bboxes).to(device)
        target["labels"] = torch.tensor(labels).to(device)
        targets.append(target)
    return images, targets