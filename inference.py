import os
import glob
import torch
import pandas as pd
from torchvision.io import read_image
from torchvision.transforms import v2
from PIL import Image, ImageDraw


class Inference():
    def __init__(self, detector, batch_size=4):
        self.detector = detector
        self.batch_size = batch_size
        self.transforms = v2.Compose([
            v2.ToTensor(),
        ])
        
    def get_number_of_batches(self, n_images: int):
        return round(n_images / self.batch_size + 0.5 - 1e-5)
        
    def load_images(
        self,
        image_paths: list[str],
        resize: int = 224,
    ) -> list[Image.Image]:
        """
        Load images from disk
        """
        images = []
        for image_path in image_paths:
            images.append(Image.open(image_path).resize((resize,resize)).convert('RGB'))
        return images
    
    def get_results(self, images):
        """
        Run images through model
        """
        self.detector.model.eval()
        results = []
        n_batches = self.get_number_of_batches(len(images))
        with torch.no_grad():
            for b in range(n_batches):
                images = self.transforms(images[b*self.batch_size: (b+1)*self.batch_size])
                images = [image.to(self.detector.device) for image in images]
                batch_results = self.detector.model(images)
                results += batch_results
        return results
    
    def get_results_df(self, image_paths: list[str], resize: int = 224):
        n_batches = self.get_number_of_batches(len(image_paths))
        df_data = []
        for b in range(n_batches):
            print("B", b)
            images = self.load_images(image_paths[b*self.batch_size: (b+1)*self.batch_size], resize=resize)
            results = self.get_results(images)
            for i, result in enumerate(results):
                boxes, labels, scores = result['boxes'], result['labels'], result['scores']
                for j in range(len(boxes)):
                    df_data.append({
                        "filename": image_paths[b*self.batch_size: (b+1)*self.batch_size][i],
                        "width": resize,
                        "height": resize,
                        "class": str(int(labels[j].item())),
                        "class_index": int(labels[j].item()),
                        "xmin": int(boxes[j][0].item()),
                        "ymin": int(boxes[j][1].item()),
                        "xmax": int(boxes[j][2].item()),
                        "ymax": int(boxes[j][3].item()),
                        "score": float(scores[j].item()),
                    })
            del images
            del results
        return pd.DataFrame(df_data)
            
    def draw_results(
        self,
        images: list[Image.Image],
        results: dict,
        score_threshold: float = 0.5,
        color: str = "#ff0000",
        show_labels: bool = True,
        make_copy=True,
    ):
        images_bboxes = []
        for i, result in enumerate(results):
            boxes, labels, scores = result['boxes'], result['labels'], result['scores']
            score_mask = scores > score_threshold
            important_boxes = boxes[score_mask]
            important_labels = labels[score_mask]
            if make_copy:
                images_bboxes.append(images[i].copy())
            else:
                images_bboxes.append(images[i])
            draw = ImageDraw.Draw(images_bboxes[i])
            for j, box in enumerate(important_boxes):
                draw.rectangle(
                    [(int(box[0]), int(box[1])), (int(box[2]), int(box[3]))],
                    outline=color,
                    width=2
                )
                if show_labels:
                    draw.text((int(box[0]), int(box[1])-15), str(important_labels[j].item()), fill=color)
        return images_bboxes


