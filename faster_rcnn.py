import torch
import torchvision
from torchvision.transforms import v2
from data import *
from metrics import *


class FasterRCNNV2Detector():
    def __init__(
        self,
        num_classes: int,
        resume: str = None,
        device: str = "cpu",
        root_dir: str = None,
    ):
        self.name = "fasterrcnn_resnet50_fpn_v2"
        self.num_classes = num_classes
        self.device = device
        self.model = self.build_model(num_classes, resume, device)
        self.root_dir = root_dir
        
    def get_weight_filename(self, epoch: int):
        return f"{self.name}_epoch{str(epoch).zfill(4)}.pth"
    
    def build_model(self, num_classes: int, resume: str, device: str = "cpu"):
        ### Instantiate model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        )

        ### Replace the head of the network
        if num_classes:
            model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, num_classes+1)

        ### Load custom weights
        if resume:
            model.load_state_dict(torch.load(resume))
        return model.to(self.device)

    def init_epoch_losses(self) -> dict:
        return {
            "loss": 0.0,
            "loss_classifier": 0.0,
            "loss_objectness": 0.0,
            "loss_box_reg": 0.0,
            "loss_rpn_box_reg": 0.0
        }

    def update_epoch_losses(self, epoch_losses: dict, loss_dict: dict):
        loss = sum(v for v in loss_dict.values())
        loss_total = loss.detach().cpu().numpy()
        loss_classifier = loss_dict['loss_classifier'].detach().cpu().numpy()
        loss_box_reg = loss_dict['loss_box_reg'].detach().cpu().numpy()
        loss_objectness = loss_dict['loss_objectness'].detach().cpu().numpy()
        loss_rpn_box_reg = loss_dict['loss_rpn_box_reg'].detach().cpu().numpy()

        epoch_losses["loss"] += loss_total
        epoch_losses["loss_classifier"] += loss_classifier
        epoch_losses["loss_box_reg"] += loss_box_reg
        epoch_losses["loss_objectness"] += loss_objectness
        epoch_losses["loss_rpn_box_reg"] += loss_rpn_box_reg

        return epoch_losses
    
    def log_epoch_metrics(self, n_epochs, epoch, epoch_metrics):
        log_items = []
        for key in epoch_metrics.keys():
            log_items.append(f"{key}={epoch_metrics[key]}")
        log_text = f"Epoch {epoch}/{n_epochs}: "
        log_text += (", ".join(log_items))
        print(f"Epoch {epoch}/{n_epochs}: {log_text}")


    def train_one_epoch(self, train_dataloader, optim) -> dict:
        # Initialize epoch losses
        epoch_losses = self.init_epoch_losses()    
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
        train_data_dir: str = None,
        train_annotation_file: str = None,
        train_transforms = None,
        val_data_dir: str = None,
        val_annotation_file: str = None,
        val_transforms = None,
        val_batch_size: int = 2,
        test_data_dir: str = None,
        test_annotation_file: str = None,
        test_transforms = None,
        test_batch_size: int = 2,
    ):
        # check if weights from previous epoch are available
        if resume is None and start_epoch and start_epoch > 0:
            resume = self.root_dir + "/" + get_weight_filename(start_epoch)

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
            collate_fn=collate_fn_coco
        )
        train_dataloader_mini = build_dataloader(
            dataset=train_dataset,
            batch_size=2,
            collate_fn=collate_fn_coco
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
                collate_fn=collate_fn_coco
            )

        # Optimizer
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.1)

        self.model.train()
        self.model.to(self.device)
        print("Start training ...")

        for epoch in range(start_epoch+1, n_epochs+1):
            # Put model into training mode
            self.model.train()
            
            # Train the model for one epoch
            epoch_metrics = self.train_one_epoch(train_dataloader, optim)
            
            # Save model weights
            if save_every and epoch > 0 and epoch % save_every == 0:
                torch.save(self.model.state_dict(), self.get_weight_filename(epoch))
                
            # Update learning rate
            if epoch > 0 and epoch % lr_step_every == 0:
                lr_scheduler.step()
                
            # Validate model
            if val_dataloader is not None:
                # Put model into evaluation mode
                self.model.eval()
                train_mAP50, train_mAP50_95 = compute_mAP_values(
                    model=self.model,
                    dataloader=train_dataloader_mini,
                    n_batches_validation=4,
                    device=device
                )
                epoch_metrics |= {
                    'train_map50': train_mAP50,
                    'train_mAP50_95': train_mAP50_95,
                }
                val_mAP50, val_mAP50_95 = compute_mAP_values(
                    model=self.model,
                    dataloader=val_dataloader,
                    n_batches_validation=4,
                    device=device
                )
                epoch_metrics |= {
                    'val_map50': val_mAP50,
                    'val_mAP50_95': val_mAP50_95,
                }
            self.log_epoch_metrics(n_epochs=n_epochs, epoch=epoch, epoch_metrics=epoch_metrics)
