# Inference with a detector

Every detector wrapper exposes an `inference()` helper:

```python
detector = FasterRCNNV2Detector(num_classes=num_classes, resume=checkpoint, device=device)
inference = detector.inference()
```

`detector.inference()` returns an `Inference` object from `/home/runner/work/pytorch-detector/pytorch-detector/inference.py`.

## Default behavior

The inference helper:

- keeps a reference to the detector instance
- uses a default batch size of `4`
- converts images to float tensors with torchvision v2 transforms
- switches the model to evaluation mode and runs under `torch.no_grad()`

## Running predictions

There are two main prediction flows.

### `get_results(images)`

Use this when you already have a list of loaded PIL images.

It returns the raw model output for each image, including:

- `boxes`
- `labels`
- `scores`

### `get_results_df(image_paths, resize=224)`

Use this when your input is a list of file paths.

For each batch, the helper:

1. loads the images from disk
2. resizes every image to a square `resize x resize`
3. runs the detector
4. flattens all detections into a Pandas DataFrame

The resulting DataFrame contains:

- `filename`
- `width`
- `height`
- `class`
- `class_index`
- `xmin`, `ymin`, `xmax`, `ymax`
- `score`

## Drawing boxes

`draw_results(...)` overlays predictions on PIL images.

Useful options:

- `score_threshold`: hides predictions below the threshold
- `color`: bounding-box and label color
- `show_labels`: toggles numeric class labels
- `make_copy`: keeps the original images unchanged when `True`

## Important inference detail

The helper reports numeric class ids, not class names. If you need human-readable labels, keep your own mapping from dataset class id to display name next to your inference script.
