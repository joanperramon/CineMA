"""A minimalist script to train a classification model on ACDC dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandSpatialCropd,
    ScaleIntensityd,
    SpatialPadd,
    Transform,
)
from safetensors.torch import save_file
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from cinema import ConvViT
from cinema.classification.train import classification_metrics
from cinema.convvit import param_groups_lr_decay
from cinema.device import get_amp_dtype_and_device
from cinema.log import get_logger
from cinema.optim import EarlyStopping, GradScaler, adjust_learning_rate

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = get_logger(__name__)


class ACDCDataset(Dataset):
    """Dataset for ACDC ED/ES frame classification."""

    def __init__(
        self,
        data_dir: Path,
        meta_df: pd.DataFrame,
        class_col: str,
        classes: list[str],
        transform: Transform,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initialize the dataset."""
        self.data_dir = data_dir
        self.meta_df = meta_df
        self.transform = transform
        self.dtype = dtype
        self.class_col = class_col
        self.classes = classes

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.meta_df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a sample."""
        row = self.meta_df.iloc[idx]
        pid = row["pid"]
        pid_dir = self.data_dir / str(pid)

        ed_image_path = pid_dir / f"{pid}_sax_ed.nii.gz"
        es_image_path = pid_dir / f"{pid}_sax_es.nii.gz"
        
        # Use nibabel (more lenient with direction cosines)
        ed_img = nib.load(str(ed_image_path))
        es_img = nib.load(str(es_image_path))
        
        # Get data arrays - nibabel returns in (x, y, z) format
        ed_image = ed_img.get_fdata()
        es_image = es_img.get_fdata()
        data = {
            "pid": pid,
            "sax_image": torch.from_numpy(np.stack([ed_image, es_image], axis=0)),  # (2, x, y, z)
            "label": torch.tensor(self.classes.index(row[self.class_col])),
        }
        data = self.transform(data)
        return data


def get_dataloaders(config: DictConfig) -> tuple[DataLoader, DataLoader]:
    """Get the dataloaders."""
    # Read data_dir from config - check both top-level and experiment level
    data_dir = None
    
    # Try top-level data_dir first (from command line override)
    if hasattr(config, "data_dir") and config.data_dir:
        data_dir = Path(config.data_dir)
        logger.info(f"Using data_dir from config (top-level): {data_dir}")
    # Try experiment.data_dir (from classification.yaml)
    elif hasattr(config, "experiment") and hasattr(config.experiment, "data_dir") and config.experiment.data_dir:
        data_dir = Path(config.experiment.data_dir)
        logger.info(f"Using data_dir from experiment config: {data_dir}")
    
    # If still not found, download from HuggingFace
    if data_dir is None:
        logger.info("No data_dir found in config. Downloading ACDC dataset from HuggingFace...")
        data_dir = Path(
            snapshot_download(repo_id="mathpluscode/ACDC", allow_patterns=["*.nii.gz", "*.csv"], repo_type="dataset")
        )
    
    logger.info(f"Loading data from: {data_dir}")
    meta_df = pd.read_csv(data_dir / "train.csv")

    # Determine classification task
    class_col = config.data.class_column
    is_binary = class_col == "hf_label"
    
    if is_binary:
        logger.info(f"Binary classification task: {class_col}")
    else:
        logger.info(f"Multi-class classification task: {class_col}")
    
    # Load predefined validation split from dataset
    val_csv = data_dir / "val.csv"
    if val_csv.exists():
        val_meta_df = pd.read_csv(val_csv)
        val_pids = val_meta_df["pid"].tolist()
        logger.info(f"Loaded predefined validation split: {len(val_pids)} patients")
    else:
        # Fallback to stratified sampling if val.csv doesn't exist
        try:
            groupby_col = "hf_label" if is_binary else class_col
            val_pids = meta_df.groupby(groupby_col).sample(n=2, random_state=0)["pid"].tolist()
            logger.warning(f"No val.csv found. Using stratified sampling by {groupby_col}")
        except ValueError as e:
            logger.warning(f"Could not stratify by {groupby_col}: {e}. Using random split.")
            val_pids = meta_df.sample(frac=0.2, random_state=0)["pid"].tolist()
    
    train_meta_df = meta_df[~meta_df["pid"].isin(val_pids)].reset_index(drop=True)
    if config.data.max_n_samples > 0:
        train_meta_df = train_meta_df.head(config.data.max_n_samples)
        logger.warning(f"Using {len(train_meta_df)} samples instead of {config.data.max_n_samples}.")
    val_meta_df = val_meta_df[val_meta_df["pid"].isin(val_pids)].reset_index(drop=True)

    print("\nTraining class distribution:")
    print(train_meta_df[class_col].value_counts())
    print()
    
    print("\nValidation class distribution:")
    print(val_meta_df[class_col].value_counts())
    print()

    patch_size_dict = {"sax": config.data.sax.patch_size}
    rotate_range_dict = {"sax": config.transform.sax.rotate_range}
    translate_range_dict = {"sax": config.transform.sax.translate_range}
    classes = config.data[class_col]
    train_transform = Compose(
        [
            RandAdjustContrastd(keys="sax_image", prob=config.transform.prob, gamma=config.transform.gamma),
            RandGaussianNoised(keys="sax_image", prob=config.transform.prob),
            ScaleIntensityd(keys="sax_image"),
            RandAffined(
                keys="sax_image",
                mode="bilinear",
                prob=config.transform.prob,
                rotate_range=tuple(r / 180 * np.pi for r in rotate_range_dict["sax"]),
                translate_range=translate_range_dict["sax"],
                scale_range=config.transform.scale_range,
                padding_mode="zeros",
                lazy=True,
            ),
            RandSpatialCropd(
                keys="sax_image",
                roi_size=patch_size_dict["sax"],
                lazy=True,
            ),
            SpatialPadd(
                keys="sax_image",
                spatial_size=patch_size_dict["sax"],
                method="end",
                lazy=True,
            ),
        ]
    )
    val_transform = Compose(
        [
            ScaleIntensityd(keys="sax_image"),
            SpatialPadd(
                keys="sax_image",
                spatial_size=patch_size_dict["sax"],
                method="end",
                lazy=True,
            ),
        ]
    )
    train_dataset = ACDCDataset(
        data_dir=data_dir / "train",
        meta_df=train_meta_df,
        transform=train_transform,
        class_col=class_col,
        classes=classes,
    )
    val_dataset = ACDCDataset(
        data_dir=data_dir / "val",  # Changed from "train" to "val"
        meta_df=val_meta_df,
        transform=val_transform,
        class_col=class_col,
        classes=classes,
    )

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=config.train.n_workers,
    )

    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        sampler=val_sampler,
        batch_size=1,
        drop_last=False,
        pin_memory=True,
        num_workers=config.train.n_workers,
    )
    return train_dataloader, val_dataloader


@hydra.main(version_base=None, config_path="", config_name="classification")
def run(config: DictConfig) -> None:
    """Entrypoint for classification training.

    Args:
        config: config loaded from yaml.
    """
    amp_dtype, device = get_amp_dtype_and_device()
    torch.manual_seed(config.seed)
    ckpt_dir = Path(config.logging.dir) / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # load dataset
    train_dataloader, val_dataloader = get_dataloaders(config=config)

    # load model
    model = ConvViT.from_pretrained(
        config=config,
        freeze=config.model.freeze_pretrained,
    )
    param_groups = param_groups_lr_decay(
        model,
        no_weight_decay_list=[],
        weight_decay=config.train.weight_decay,
        layer_decay=config.train.layer_decay,
    )
    model.set_grad_ckpt(config.grad_ckpt)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"number of parameters: {n_params:,}")
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of trainable parameters: {n_trainable_params:,}")

    # init optimizer
    logger.info("Initializing optimizer.")
    optimizer = torch.optim.AdamW(param_groups, lr=config.train.lr, betas=config.train.betas)
    loss_scaler = GradScaler()

    # Compute class weights for imbalanced dataset
    train_labels = []
    for batch in train_dataloader:
        train_labels.extend(batch["label"].tolist())
    class_counts = torch.bincount(torch.tensor(train_labels))
    
    # Calculate imbalance ratio (max/min class count)
    imbalance_ratio = class_counts.max().item() / class_counts.min().item()
    
    # Get class weight settings from config, with defaults
    use_class_weights = config.train.get('use_class_weights', True)  # Default: enabled
    class_weight_threshold = config.train.get('class_weight_threshold', 1.2)  # Default: 1.2:1 ratio
    
    if use_class_weights and imbalance_ratio > class_weight_threshold:
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum() * len(class_weights)  # Normalize
        class_weights = class_weights.to(device)
        logger.info(f"Class imbalance detected (ratio {imbalance_ratio:.2f}). Applying class weights: {class_weights.tolist()}")
    else:
        class_weights = None
        if not use_class_weights:
            logger.info(f"Class weights disabled in config. Using uniform weights.")
        else:
            logger.info(f"Dataset is balanced (ratio {imbalance_ratio:.2f} <= {class_weight_threshold}). Using uniform weights.")
    
    # train
    logger.info("Start training.")
    # Get mode from config, default to 'max' for classification metrics
    early_stop_mode = config.train.early_stopping.get('mode', 'max')
    early_stop = EarlyStopping(
        min_delta=config.train.early_stopping.min_delta,
        patience=config.train.early_stopping.patience,
        mode=early_stop_mode,
    )
    logger.info(f"Early stopping: metric={config.train.early_stopping.metric}, mode={early_stop_mode}")
    n_samples = 0
    for epoch in range(config.train.n_epochs):
        optimizer.zero_grad()
        model.train()
        for i, batch in enumerate(train_dataloader):
            lr = adjust_learning_rate(
                optimizer=optimizer,
                step=i / len(train_dataloader) + epoch,
                warmup_steps=config.train.n_warmup_epochs,
                max_n_steps=config.train.n_epochs,
                lr=config.train.lr,
                min_lr=config.train.min_lr,
            )
            with torch.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                logits = model({"sax": batch["sax_image"].to(device)})
                label = batch["label"].long().to(device)
                loss = F.cross_entropy(
                    logits,
                    label,
                    weight=class_weights,  # Apply class weights if dataset is imbalanced (None = uniform)
                    label_smoothing=0.1,
                )
                metrics = {"train_cross_entropy_loss": loss.item()}

            grad_norm = loss_scaler(
                loss=loss,
                optimizer=optimizer,
                clip_grad=config.train.clip_grad,
                parameters=model.parameters(),
                update_grad=True,
            )
            optimizer.zero_grad()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            n_samples += batch["sax_image"].shape[0]
            metrics.update({"grad_norm": grad_norm.item(), "lr": lr})
            metrics = {k: f"{v:.2e}" for k, v in metrics.items()}
            logger.info(f"Epoch {epoch}, step {i}, metrics: {metrics}.")

        if (ckpt_dir is None) or ((epoch + 1) % config.train.eval_interval != 0):
            continue

        # evaluate model
        logger.info(f"Start evaluating model at epoch {epoch}.")
        model.eval()
        true_labels = []
        pred_logits = []
        for _, batch in enumerate(val_dataloader):
            with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                logits = model({"sax": batch["sax_image"].to(device)})
                pred_logits.append(logits)
                true_labels.append(batch["label"])
        
        pred_logits = torch.cat(pred_logits, dim=0).cpu().to(dtype=torch.float32)
        pred_probs = F.softmax(pred_logits, dim=1).numpy()
        true_labels = torch.cat(true_labels, dim=0).cpu().to(dtype=torch.float32).numpy()
        pred_labels = torch.argmax(pred_logits, dim=1).numpy()
        
        # Compute comprehensive classification metrics (ROC-AUC, MCC, F1, accuracy, etc.)
        metrics = classification_metrics(
            true_labels=true_labels,
            pred_labels=pred_labels,
            pred_probs=pred_probs,
        )
        
        # Add val_ prefix for all metrics
        val_metrics = {f"val_{k}": v for k, v in metrics.items()}
        val_metrics_formatted = {k: f"{v:.2e}" for k, v in val_metrics.items()}
        logger.info(f"Validation metrics: {val_metrics_formatted}.")

        # save model checkpoint
        ckpt_path = ckpt_dir / f"ckpt_{epoch}.safetensors"
        save_file(model.state_dict(), ckpt_path)
        logger.info(f"Saved checkpoint of epoch {epoch} at {ckpt_path} after {n_samples} samples.")

        # early stopping - use the configured metric
        early_stop_metric_name = config.train.early_stopping.metric
        if early_stop_metric_name not in val_metrics:
            logger.warning(
                f"Configured early stopping metric '{early_stop_metric_name}' not found in computed metrics. "
                f"Available metrics: {list(val_metrics.keys())}. Falling back to 'val_roc_auc'."
            )
            early_stop_metric_name = "val_roc_auc"
        early_stop_metric_value = val_metrics[early_stop_metric_name]
        early_stop.update(early_stop_metric_value)
        if early_stop.should_stop:
            logger.info(
                f"Met early stopping criteria with {config.train.early_stopping.metric} = "
                f"{early_stop.best_metric} and patience {early_stop.patience_count}, breaking."
            )
            break


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
