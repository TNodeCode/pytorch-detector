# PyTorch Object Detection

This is a wrapper library for the PyTorch object detection models. The main idea of this library is to make training, validation and inference with PyTorch's object detection models as simple as possible.

## Available models

There are six models and their subtypes that you can use with this library:

- Faster R-CNN
- Faster R-CNN V2
- FCOS
- RetinaNet
- SSD
- SSDlite

For more details about these models see the <a href="https://pytorch.org/vision/stable/models.html#object-detection" target="_blank">official PyTorch Documentation</a>.

## Training models

First you need to import the necessary libraries:

```python
import os
import torch
from torchvision.transforms import v2

from detectors import *
```

Each model is wrapped by a class that provides some high level functions for training the model. Ypu can see all the available classes in the `detectors.py`. For training a model you first need to create an instance of the wrapper class. For the Faster R-CNN V2 model this works like this:

```python
detector = FasterRCNNV2Detector(
    num_classes=num_classes,    # The number of classes in your dataset
    device=device,              # Device (cpu or cuda)
    root_dir=root_dir,          # Directory from which this code is run from
)
```

The next step is to call the `train` method of the detector.

```python
detector.train(
    n_epochs = 50,                                  # numper of epochs the model is trained
    lr = 1e-3,                                      # learning rate
    batch_size = 16,                                # batch size
    start_epoch = 0,                                # start epoch (setting this to another value than 0 will only change the logs that are produced)
    resume = None,                                  # path to a checkpoint file (None means the model is trained with the COCO weights)
    save_every = 10,                                # save checkpoints every x epochs (files are stored in the log directory)
    lr_step_every = 10,                             # update the learning rate every x epochs
    num_classes = num_classes,                      # number of classes
    device=device,                                  # device that should be used for training (cpu or cuda)
    log_dir=os.path.join(root_dir, "logs", dataset_name, detector.name),    # here the log files are stored
    train_data_dir = train_data_dir,                # directory where the training data is stored
    train_annotation_file = train_annotation_file,  # path to the COCO annotation file for the training images
    train_transforms = train_transforms,            # image augmentation for the training images
    val_data_dir = val_data_dir,                    # directory where the validation data is stored
    val_annotation_file = val_annotation_file,      # path to the COCO annotation file for the validation images
    val_transforms = val_transforms,                # image augmentation for the validation images
    val_batch_size=2,                               # number of batches that are used for the validation process
    n_batches_validation=2,                         # number of images per batch that are used for the validation process
    test_data_dir = None,                           # directory where the test data is stored
    test_annotation_file = None,                    # path to the COCO annotation file for the test images
    test_transforms = None,                         # image augmentation for the test images
)
```

## Inference

Making predictions on images with your trained model is also easy with this library. Lets assume your checkpoint is saved at `"checkpoints/epoch300.pth"`. Then you can perform inference on a set of images using the following code:

```python
import glob

# Create a detector object
detector = FasterRCNNV2Detector(
    num_classes=10,
    resume="checkpoints/epoch300.pth",
    device="cuda",
)

# Create an inference object
inference = detector.inference()

# Load all JPG files from the 'images' directory
image_paths = glob.glob("images/*.jpg")

# Get a Pandas DataFrame containing the perdictions for all images
df = inference.get_results_df(
    image_paths=image_paths,    # list of image paths
    resize=512                  # size of the images
)

# Save DataFrame as CSV file
df.to_csv(csv_name, index=False)
```
