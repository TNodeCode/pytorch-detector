# Dataset organization

The training code expects a COCO-style dataset:

- image files live in a directory that you pass as `train_data_dir` or `val_data_dir`
- annotations live in a COCO JSON file that you pass as `train_annotation_file` or `val_annotation_file`
- each image entry in the annotation file must match a file inside the corresponding image directory

A practical layout looks like this:

```text
project-root/
├── datasets/
│   └── my_dataset/
│       ├── train/
│       │   ├── image_0001.jpg
│       │   ├── image_0002.jpg
│       │   └── ...
│       ├── val/
│       │   ├── image_0101.jpg
│       │   ├── image_0102.jpg
│       │   └── ...
│       └── annotations/
│           ├── train.json
│           └── val.json
└── train.py
```

## What the loader reads

`build_coco_dataset(...)` uses `torchvision.datasets.CocoDetection`, so the JSON file must follow the COCO detection schema.

During training, `extract_images_targets(...)` converts each annotation into PyTorch detection targets:

- `bbox` is expected in COCO `xywh` format and is converted to `xyxy`
- `category_id` is converted to the training label with `category_id + 1`
- label `0` is therefore reserved for the background class

## Class indexing rule

Set `num_classes` to the number of foreground classes in your dataset.

Because the code shifts every `category_id` by `+1`, the safest setup is to keep annotation class ids dense and zero-based:

- first class -> `category_id: 0`
- second class -> `category_id: 1`
- ...
- last class -> `category_id: num_classes - 1`

## Split handling

The current training loop always requires a training split and can optionally use a validation split:

- `train_data_dir` + `train_annotation_file` are required to train
- `val_data_dir` + `val_annotation_file` enable validation mAP logging
- test-related arguments exist on `train(...)`, but the current implementation does not consume them yet
