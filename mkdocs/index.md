# PyTorch Detector Documentation

This directory collects the project-specific usage notes that are only briefly covered in the root README.

- [Dataset organization](dataset.md) explains the folder layout and annotation assumptions expected by `build_coco_dataset(...)` and `extract_images_targets(...)`.
- [Training a detector](training.md) shows how the detector wrappers are instantiated and how `AbstractDetector.train(...)` uses the dataset inputs, logs, checkpoints, and validation split.
- [Inference with a detector](inference.md) explains what `detector.inference()` returns and how batched prediction and visualization work.

The documentation in this folder reflects the current code in `/home/runner/work/pytorch-detector/pytorch-detector/data.py`, `/home/runner/work/pytorch-detector/pytorch-detector/detectors.py`, and `/home/runner/work/pytorch-detector/pytorch-detector/inference.py`.
