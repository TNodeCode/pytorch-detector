# Training a detector

All detector wrappers live in `/home/runner/work/pytorch-detector/pytorch-detector/detectors.py`. They expose the same high-level `train(...)` entry point through `AbstractDetector`.

## 1. Pick a detector class

Examples include:

- `FasterRCNNDetector`
- `FasterRCNNV2Detector`
- `RetinaNetResNet50FPNDetector`
- `FCOSResNet50FPNDetector`
- `SSD300VGG16Detector`
- `DINOv2HungarianDetector`

Create the detector with the number of foreground classes and the device you want to use.

## 2. Prepare transforms and paths

The `train(...)` method expects:

- a training image directory
- a training COCO annotation file
- optional validation image and annotation paths
- torchvision v2 transforms for the training and validation sets
- a `log_dir` where epoch logs and checkpoints should be written

## 3. Call `train(...)`

The training loop performs these steps:

1. rebuilds the model head for `num_classes`
2. constructs a COCO training dataloader
3. optionally constructs a validation dataloader
4. trains for `n_epochs`
5. saves checkpoints every `save_every` epochs when enabled
6. logs metrics to `epochs.yaml` inside `log_dir`
7. computes train/validation mAP values when a validation split is provided

## Key training arguments

- `n_epochs`: total number of epochs to run
- `lr`: SGD learning rate
- `batch_size`: batch size for the training dataloader
- `resume`: checkpoint path to load model weights from
- `save_every`: checkpoint frequency
- `lr_step_every`: epoch interval used before stepping the learning-rate scheduler
- `log_dir`: output directory for logs and checkpoints
- `n_batches_validation`: number of batches evaluated when computing mAP metrics

## Logging and checkpoints

Training creates the log directory if it does not exist.

Inside that directory, the current implementation writes:

- `epochs.yaml` with one record per epoch
- checkpoint files named like `<detector-name>_epoch0001.pth`

Each checkpoint stores:

- `model_state`
- `optim_state`
- `scheduler_state`
- the detector name and class count
- the metrics collected for that epoch

## Validation behavior

Validation only runs when `val_data_dir` is provided.

When validation is enabled, the trainer computes:

- `train_map50`
- `train_mAP50_95`
- `val_map50`
- `val_mAP50_95`

The training loop also builds a small non-shuffled training dataloader for the train-side metric snapshot.
