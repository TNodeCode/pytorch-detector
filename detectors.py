import os
import torch
import torchvision
from torchvision.transforms import v2
from data import *
from metrics import *
from datetime import datetime
import logger
import inference
from feature_extractors import DINOv2ViT, DINOv2ViTBackbone, DINOv2ConvNextBackbone
from hungarian_head import HungarianDetectionHead, DINOv2HungarianDetectionModel


class AbstractDetector():
    def __init__(
        self,
        name: str,
        num_classes: int,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        log_dir: str = None
    ):
        self.name = name
        self.num_classes = num_classes
        self.device = device
        self.model = self.build_model(num_classes, resume, device)
        if resume:
            self.load_weights(resume)
        if root_dir is None:
            self.root_dir = os.getcwd()
        else:
            self.root_dir = root_dir
        self.log_dir = log_dir
        
    def inference(self):
        return inference.Inference(detector=self)
        
    def get_loss_names() -> list[str]:
        raise NotImplementedError("this function needs to be implemented")
        
    def get_weight_filename(self, epoch: int):
        return f"{self.name}_epoch{str(epoch).zfill(4)}.pth"
    
    def load_pretrained_model(self):
        raise NotImplementedError("this function needs to be implemented")
        
    def load_weights(self, filepath):
        self.model.load_state_dict(
            torch.load(filepath, map_location=torch.device(self.device))["model_state"]
        )
        
    def replace_head(self, num_classes: int):
        raise NotImplementedError("this function needs to be implemented")
    
    def build_model(self, num_classes: int = None, resume: str = None, device: str = "cpu"):
        ### Instantiate model
        model = self.load_pretrained_model()

        ### Replace the head of the network
        if num_classes:
            self.replace_head(model, num_classes)
            
        return model.to(self.device)
            
    def create_log_dir(self):
        if self.log_dir is None:
            log_dir = os.path.join(self.root_dir, "logs", self.name)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            n_models = len(os.listdir(log_dir))
            self.log_dir = os.path.join(log_dir, f"training_{n_models+1}")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)  

    def init_epoch_losses(self) -> dict:
        loss_dict = {"loss": 0.0}
        for k in self.get_loss_names():
            if k[0:5] != "loss_":
                k = "loss_" + k
            loss_dict |= {k: 0.0}
        return loss_dict

    def update_epoch_losses(self, epoch_losses: dict, loss_dict: dict):
        loss_total = (sum(v for v in loss_dict.values())).detach().cpu().numpy()
        epoch_losses["loss"] += loss_total
        for k in self.get_loss_names():
            loss_k = loss_dict[k].detach().cpu().numpy()
            if k[0:5] != "loss_":
                k = "loss_" + k
            epoch_losses[k] += loss_k
        return epoch_losses
    
    def log_epoch_metrics(self, n_epochs, epoch, epoch_metrics):
        epoch_metrics |= {"epoch": epoch}
        log_items = []
        for key in epoch_metrics.keys():
            log_items.append(f"{key}={epoch_metrics[key]}")
        log_text = f"Epoch {epoch}/{n_epochs}: "
        log_text += (", ".join(log_items))
        # Save epoch logs on disk
        logger.log_epoch(os.path.join(self.log_dir, "epochs.yaml"), epoch_metrics)
        print(f"Epoch {epoch}/{n_epochs}: {log_text}")

    def train_one_epoch(self, train_dataloader, optim) -> dict:
        # Initialize epoch losses
        epoch_losses = self.init_epoch_losses()
        self.model.train()
        for data in train_dataloader:
            # Get data
            images, targets = extract_images_targets(data, device=self.device)
            # Compute loss of batch
            loss_dict = self.model(images, targets)
            epoch_losses = self.update_epoch_losses(epoch_losses, loss_dict)
            loss = sum(v for v in loss_dict.values())
            # Backpropagation
            optim.zero_grad()
            loss.backward()
            optim.step()
        return epoch_losses

    def train(
        self,
        n_epochs: int,
        lr: float,
        batch_size: int,
        start_epoch: int = 0,
        resume: str = None,
        save_every: int = None,
        lr_step_every: int = 20,
        num_classes = None,
        device="cpu",
        log_dir: str = None,
        train_data_dir: str = None,
        train_annotation_file: str = None,
        train_transforms = None,
        val_data_dir: str = None,
        val_annotation_file: str = None,
        val_transforms = None,
        val_batch_size: int = 2,
        n_batches_validation: int = 2,
        test_data_dir: str = None,
        test_annotation_file: str = None,
        test_transforms = None,
        test_batch_size: int = 2,
    ):
        # Set log directory
        if log_dir is not None:
            self.log_dir = log_dir
        
        # check if weights from previous epoch are available
        if resume is None and start_epoch and start_epoch > 0:
            resume = self.root_dir + "/" + self.get_weight_filename(start_epoch)

        # Instantiate model
        self.model = self.build_model(num_classes=num_classes, resume=resume, device=device)
        
        # Build train dataloader
        train_dataset = build_coco_dataset(
            root=train_data_dir,
            annFile=train_annotation_file,
            transform=train_transforms
        )
        train_dataloader = build_dataloader(
            dataset=train_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn_coco,
            shuffle=True,
            drop_last=True,
        )
        train_dataloader_mini = build_dataloader(
            dataset=train_dataset,
            batch_size=2,
            collate_fn=collate_fn_coco,
            shuffle=False,
            drop_last=False,
        )
        
        # Build validation dataloader
        val_dataset = None
        val_dataloader = None
        if val_data_dir is not None:
            val_dataset = build_coco_dataset(
                root=val_data_dir,
                annFile=val_annotation_file,
                transform=val_transforms
            )
            val_dataloader = build_dataloader(
                dataset=val_dataset,
                batch_size=val_batch_size,
                collate_fn=collate_fn_coco,
                shuffle=False,
                drop_last=False,
            )

        # Optimizer
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.1)
        
        # Build log directory
        self.create_log_dir()
        print(f"Logging training at {self.log_dir}")

        self.model.to(self.device)
        print("Start training ...")

        for epoch in range(start_epoch+1, n_epochs+1):
            # Put model into training mode
            self.model.train()
            
            # Train the model for one epoch
            print("Train ...")
            epoch_metrics = {"learning_rate": lr_scheduler.get_last_lr()[0],
                             "lr_step_every": lr_step_every,
                             "optim": str(type(optim)),
                             "scheduler": str(type(lr_scheduler)),
                             "epoch_start": datetime.now().astimezone().isoformat(),
                             "batch_size": batch_size,
                             "val_batch_size": val_batch_size,
                             "n_batches_validation": n_batches_validation}
            epoch_metrics |= self.train_one_epoch(train_dataloader, optim)          
            epoch_metrics |= {"epoch_end": datetime.now().astimezone().isoformat()}
            
            # Save model weights
            if save_every and epoch > 0 and epoch % save_every == 0:
                torch.save(
                    {"name": self.name,
                     "num_classes": self.num_classes,
                     "model_state": self.model.state_dict(),
                     "optim_state": optim.state_dict(),
                     "scheduler_state": lr_scheduler.state_dict(),
                     **epoch_metrics,
                    },
                    os.path.join(self.log_dir, self.get_weight_filename(epoch))
                )
                
            # Update learning rate
            if epoch > 0 and epoch % lr_step_every == 0:
                lr_scheduler.step()
                
            # Validate model
            if val_dataloader is not None:
                print("Validating ...")
                # Put model into evaluation mode
                self.model.eval()
                train_mAP50, train_mAP50_95 = compute_mAP_values(
                    model=self.model,
                    dataloader=train_dataloader_mini,
                    n_batches_validation=n_batches_validation,
                    device=device
                )
                epoch_metrics |= {
                    'train_map50': train_mAP50,
                    'train_mAP50_95': train_mAP50_95,
                }
                val_mAP50, val_mAP50_95 = compute_mAP_values(
                    model=self.model,
                    dataloader=val_dataloader,
                    n_batches_validation=n_batches_validation,
                    device=device
                )
                epoch_metrics |= {
                    'val_map50': val_mAP50,
                    'val_mAP50_95': val_mAP50_95,
                }
            self.log_epoch_metrics(n_epochs=n_epochs, epoch=epoch, epoch_metrics=epoch_metrics)

        
class FasterRCNNDetector(AbstractDetector):
    def __init__(
        self,
        num_classes: int = None,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
    ):
        super().__init__(
            name="fasterrcnn_resnet50_fpn",
            num_classes=num_classes,
            resume=resume,
            device=device,
            root_dir=root_dir
        )
        
    def get_loss_names(self) -> list[str]:
        return ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
    
    def load_pretrained_model(self):
        return torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        
    def replace_head(self, model, num_classes: int):
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            model.roi_heads.box_predictor.cls_score.in_features, num_classes+1
        )

        
class FasterRCNNV2Detector(AbstractDetector):
    def __init__(
        self,
        num_classes: int = None,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
    ):
        super().__init__(
            name="fasterrcnn_resnet50_fpn_v2",
            num_classes=num_classes,
            resume=resume,
            device=device,
            root_dir=root_dir
        )
        
    def get_loss_names(self) -> list[str]:
        return ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
    
    def load_pretrained_model(self):
        return torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        )
        
    def replace_head(self, model, num_classes: int):
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            model.roi_heads.box_predictor.cls_score.in_features, num_classes+1
        )

        
class FasterRCNNMobileNetV3LargeDetector(AbstractDetector):
    def __init__(
        self,
        num_classes: int = None,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
    ):
        super().__init__(
            name="fasterrcnn_mobilenet_v3_large_fpn",
            num_classes=num_classes,
            resume=resume,
            device=device,
            root_dir=root_dir
        )
        
    def get_loss_names(self) -> list[str]:
        return ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
    
    def load_pretrained_model(self):
        return torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        )
        
    def replace_head(self, model, num_classes: int):
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            model.roi_heads.box_predictor.cls_score.in_features, num_classes+1
        )

        
class FasterRCNNMobileNetV3Large320Detector(AbstractDetector):
    def __init__(
        self,
        num_classes: int = None,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
    ):
        super().__init__(
            name="fasterrcnn_mobilenet_v3_large_320_fpn",
            num_classes=num_classes,
            resume=resume,
            device=device,
            root_dir=root_dir
        )
        
    def get_loss_names(self) -> list[str]:
        return ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
    
    def load_pretrained_model(self):
        return torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        )
        
    def replace_head(self, model, num_classes: int):
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            model.roi_heads.box_predictor.cls_score.in_features, num_classes+1
        )


class RetinaNetResNet50FPNDetector(AbstractDetector):
    def __init__(
        self,
        num_classes: int = None,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
    ):
        super().__init__(
            name="retinanet_resnet50_fpn",
            num_classes=num_classes,
            resume=resume,
            device=device,
            root_dir=root_dir
        )
        
    def get_loss_names(self) -> list[str]:
        return ["classification", "bbox_regression"]
    
    def load_pretrained_model(self):
        return torchvision.models.detection.retinanet_resnet50_fpn(
            weights=torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT
        )
        
    def replace_head(self, model, num_classes: int):
        # First get some parameters of the original classification head
        in_channels = model.head.classification_head.cls_logits.in_channels
        num_anchors = model.head.classification_head.num_anchors

        # Build a new classification head
        model.head = torchvision.models.detection.retinanet.RetinaNetHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes+1,
        )


class RetinaNetResNet50FPNV2Detector(AbstractDetector):
    def __init__(
        self,
        num_classes: int = None,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
    ):
        super().__init__(
            name="retinanet_resnet50_fpn_v2",
            num_classes=num_classes,
            resume=resume,
            device=device,
            root_dir=root_dir
        )
        
    def get_loss_names(self) -> list[str]:
        return ["classification", "bbox_regression"]
    
    def load_pretrained_model(self):
        return torchvision.models.detection.retinanet_resnet50_fpn_v2(
            weights=torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        )
        
    def replace_head(self, model, num_classes: int):
        # First get some parameters of the original classification head
        in_channels = model.head.classification_head.cls_logits.in_channels
        num_anchors = model.head.classification_head.num_anchors

        # Build a new classification head
        model.head = torchvision.models.detection.retinanet.RetinaNetHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes+1,
        )
    

class FCOSResNet50FPNDetector(AbstractDetector):
    def __init__(
        self,
        num_classes: int = None,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
    ):
        super().__init__(
            name="fcos_resnet50_fpn",
            num_classes=num_classes,
            resume=resume,
            device=device,
            root_dir=root_dir
        )
        
    def get_loss_names(self) -> list[str]:
        return ["classification", "bbox_regression", "bbox_ctrness"]
    
    def load_pretrained_model(self):
        return torchvision.models.detection.fcos_resnet50_fpn(
            weights=torchvision.models.detection.FCOS_ResNet50_FPN_Weights.DEFAULT
        )
        
    def replace_head(self, model, num_classes: int):
        # First get some parameters of the original classification head
        in_channels = model.head.classification_head.cls_logits.in_channels
        num_anchors = model.head.classification_head.num_anchors
        num_convs = len(model.head.classification_head.conv) // 3

        # Build a new classification head
        model.head = torchvision.models.detection.fcos.FCOSHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes+1,
            num_convs=num_convs
        )
    

class SSD300VGG16Detector(AbstractDetector):
    def __init__(
        self,
        num_classes: int = None,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
    ):
        super().__init__(
            name="ssd300_vgg16",
            num_classes=num_classes,
            resume=resume,
            device=device,
            root_dir=root_dir
        )
        
    def get_loss_names(self) -> list[str]:
        return ["classification", "bbox_regression"]
    
    def load_pretrained_model(self):
        return torchvision.models.detection.ssd300_vgg16(
            weights=torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT
        )
        
    def replace_head(self, model, num_classes: int):
        # First get some parameters of the original classification head
        in_channels = [512,1024,512,256,256,256]
        num_anchors = [4,6,6,6,4,4]
        # Build a new classification head
        model.head = torchvision.models.detection.ssd.SSDHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes+1,
        )
    

class SSDLite320MobileNetV3LargeDetector(AbstractDetector):
    def __init__(
        self,
        num_classes: int = None,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
    ):
        super().__init__(
            name="ssdlite320_mobilenet_v3_large",
            num_classes=num_classes,
            resume=resume,
            device=device,
            root_dir=root_dir
        )
        
    def get_loss_names(self) -> list[str]:
        return ["classification", "bbox_regression"]
    
    def load_pretrained_model(self):
        return torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        )
        
    def replace_head(self, model, num_classes: int):
        # First get some parameters of the original classification head
        in_channels = [672,480,512,256,256,128]
        num_anchors = [6,6,6,6,6,6]
        # Build a new classification head
        model.head = torchvision.models.detection.ssd.SSDHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes+1,
        )


class DINOv2ViTRetinaNetDetector(AbstractDetector):
    """RetinaNet detector with a DINOv2 ViT feature extractor backbone.

    The DINOv2 ViT last-layer features are projected and downsampled into a
    five-level feature pyramid, which is consumed by a RetinaNet head.

    Args:
        num_classes (int | None): Number of foreground classes.  When
            provided the detection head is replaced to match.
        resume (str | None): Path to a checkpoint to resume from.
        device (str): ``"cpu"`` or a CUDA device string.
        root_dir (str | None): Root directory used for logging.
        finetuning (bool): If ``True`` the DINOv2 backbone weights are
            updated during training.  Defaults to ``False`` (frozen).
        model_name (str | None): HuggingFace model identifier for pretrained
            DINOv2 weights (e.g. ``'facebook/dinov2-base'``).  When ``None``
            the backbone is randomly initialised.
    """

    def __init__(
        self,
        num_classes: int = None,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        finetuning: bool = False,
        model_name: str = None,
    ):
        self.finetuning = finetuning
        self.model_name = model_name
        super().__init__(
            name="dinov2_vit_retinanet",
            num_classes=num_classes,
            resume=resume,
            device=device,
            root_dir=root_dir,
        )

    def get_loss_names(self) -> list[str]:
        return ["classification", "bbox_regression"]

    def load_pretrained_model(self):
        backbone = DINOv2ViTBackbone(
            model_name=self.model_name,
            out_channels=256,
            finetuning=self.finetuning,
        )
        return torchvision.models.detection.RetinaNet(
            backbone=backbone,
            num_classes=91,
        )

    def replace_head(self, model, num_classes: int):
        in_channels = model.head.classification_head.cls_logits.in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head = torchvision.models.detection.retinanet.RetinaNetHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes + 1,
        )


class DINOv2ConvNextRetinaNetDetector(AbstractDetector):
    """RetinaNet detector with a ConvNext feature extractor backbone.

    All four ConvNext stages are passed through an FPN to produce a
    five-level (4 FPN + max-pool) feature pyramid that feeds the RetinaNet
    head.

    Args:
        num_classes (int | None): Number of foreground classes.  When
            provided the detection head is replaced to match.
        resume (str | None): Path to a checkpoint to resume from.
        device (str): ``"cpu"`` or a CUDA device string.
        root_dir (str | None): Root directory used for logging.
        finetuning (bool): If ``True`` the ConvNext backbone weights are
            updated during training.  Defaults to ``False`` (frozen).
        model_name (str | None): HuggingFace model identifier for pretrained
            ConvNext weights (e.g. ``'facebook/convnext-base-224'``).  When
            ``None`` the backbone is randomly initialised.
    """

    def __init__(
        self,
        num_classes: int = None,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        finetuning: bool = False,
        model_name: str = None,
    ):
        self.finetuning = finetuning
        self.model_name = model_name
        super().__init__(
            name="dinov2_convnext_retinanet",
            num_classes=num_classes,
            resume=resume,
            device=device,
            root_dir=root_dir,
        )

    def get_loss_names(self) -> list[str]:
        return ["classification", "bbox_regression"]

    def load_pretrained_model(self):
        backbone = DINOv2ConvNextBackbone(
            model_name=self.model_name,
            out_channels=256,
            finetuning=self.finetuning,
        )
        return torchvision.models.detection.RetinaNet(
            backbone=backbone,
            num_classes=91,
        )

    def replace_head(self, model, num_classes: int):
        in_channels = model.head.classification_head.cls_logits.in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head = torchvision.models.detection.retinanet.RetinaNetHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes + 1,
        )


class DINOv2HungarianDetector(AbstractDetector):
    """DINOv2 detector with a DETR-style Hungarian-matching detection head.

    Combines a frozen (or optionally fine-tuned) :class:`DINOv2ViT` backbone
    with a :class:`~hungarian_head.HungarianDetectionHead` that uses the
    Hungarian algorithm to assign predicted queries to ground-truth objects
    during training.

    The model is trained end-to-end with a set-based loss consisting of three
    terms: cross-entropy classification loss, L1 bounding-box regression loss,
    and GIoU bounding-box regression loss.

    Args:
        num_classes (int | None): Number of foreground object classes.
        resume (str | None): Path to a checkpoint file to load weights from.
        device (str): ``'cpu'`` or a CUDA device string (e.g. ``'cuda:0'``).
        root_dir (str | None): Root directory for logging.
        finetuning (bool): If ``True`` the DINOv2 backbone weights are updated
            during training.  Defaults to ``False`` (backbone frozen).
        model_name (str | None): HuggingFace model identifier for pretrained
            DINOv2 weights (e.g. ``'facebook/dinov2-base'``).  When ``None``
            the backbone is randomly initialised.
        hidden_dim (int): Transformer / embedding dimensionality.  Default: 256.
        num_queries (int): Number of learnable object queries.  Default: 100.
        nhead (int): Number of attention heads per transformer decoder layer.
            Default: 8.
        num_decoder_layers (int): Depth of the transformer decoder.
            Default: 6.
        num_feature_levels (int): Number of multi-scale feature levels used by
            the deformable decoder neck. Default: 4.
        score_threshold (float): Minimum foreground score to keep during
            inference.  Default: 0.5.
    """

    def __init__(
        self,
        num_classes: int = None,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
        finetuning: bool = False,
        model_name: str = None,
        hidden_dim: int = 256,
        num_queries: int = 100,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        num_feature_levels: int = 4,
        score_threshold: float = 0.5,
    ):
        # Set attributes before calling super().__init__ because
        # AbstractDetector.__init__ immediately calls build_model → load_pretrained_model.
        self.finetuning = finetuning
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.nhead = nhead
        self.num_decoder_layers = num_decoder_layers
        self.num_feature_levels = num_feature_levels
        self.score_threshold = score_threshold
        super().__init__(
            name="dinov2_hungarian",
            num_classes=num_classes,
            resume=resume,
            device=device,
            root_dir=root_dir,
        )

    def get_loss_names(self) -> list[str]:
        return ["classification", "bbox", "giou"]

    def load_pretrained_model(self):
        backbone = DINOv2ViT(
            model_name=self.model_name,
            finetuning=self.finetuning,
            layers=[],
            layer_norm=False,
        )
        in_channels = backbone.config.hidden_size
        head = HungarianDetectionHead(
            in_channels=in_channels,
            num_classes=91,  # COCO default; replaced by replace_head when num_classes is set
            hidden_dim=self.hidden_dim,
            num_queries=self.num_queries,
            nhead=self.nhead,
            num_decoder_layers=self.num_decoder_layers,
            num_feature_levels=self.num_feature_levels,
            score_threshold=self.score_threshold,
        )
        return DINOv2HungarianDetectionModel(backbone=backbone, head=head)

    def replace_head(self, model, num_classes: int):
        in_channels = model.backbone.config.hidden_size
        model.head = HungarianDetectionHead(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_dim=self.hidden_dim,
            num_queries=self.num_queries,
            nhead=self.nhead,
            num_decoder_layers=self.num_decoder_layers,
            num_feature_levels=self.num_feature_levels,
            score_threshold=self.score_threshold,
        )
