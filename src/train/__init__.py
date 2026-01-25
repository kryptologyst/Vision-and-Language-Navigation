"""Training and evaluation loops for Vision-and-Language Navigation.

This module provides comprehensive training and evaluation functionality
with proper checkpointing, logging, and metrics tracking.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
import wandb
from tensorboard import SummaryWriter

from ..models import CLIPVLNModel, AdvancedVLNModel
from ..data import VLNDataset, create_dataloader
from ..eval import VLNMetrics, evaluate_vln_model, print_metrics
from ..utils import get_device, set_seed, save_checkpoint, load_checkpoint, EarlyStopping


logger = logging.getLogger(__name__)


class VLNTrainer:
    """Trainer class for Vision-and-Language Navigation models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        """Initialize VLN trainer.
        
        Args:
            model: VLN model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            config: Training configuration
            device: Device to train on
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device or get_device()
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup loss functions
        self.criterion = self._setup_loss_functions()
        
        # Setup mixed precision training
        self.scaler = GradScaler() if config.get("mixed_precision", False) else None
        
        # Setup logging
        self.writer = None
        self.use_wandb = config.get("logging", {}).get("use_wandb", False)
        self.use_tensorboard = config.get("logging", {}).get("use_tensorboard", True)
        
        if self.use_tensorboard:
            log_dir = config.get("logging", {}).get("log_dir", "./logs")
            self.writer = SummaryWriter(log_dir)
        
        if self.use_wandb:
            wandb.init(
                project=config.get("logging", {}).get("wandb_project", "vln-navigation"),
                config=config
            )
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get("training", {}).get("early_stopping_patience", 20)
        )
        
        # Training state
        self.current_epoch = 0
        self.best_metrics = {}
        self.training_history = []
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer.
        
        Returns:
            Configured optimizer
        """
        training_config = self.config.get("training", {})
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config.get("learning_rate", 1e-4),
            weight_decay=training_config.get("weight_decay", 1e-5)
        )
        
        return optimizer
    
    def _setup_loss_functions(self) -> Dict[str, nn.Module]:
        """Setup loss functions.
        
        Returns:
            Dictionary of loss functions
        """
        losses = {
            "action_loss": nn.CrossEntropyLoss(ignore_index=0),  # Ignore padding
            "nav_state_loss": nn.BCELoss(),
            "planning_loss": nn.MSELoss()
        }
        
        return losses
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_action_loss = 0.0
        total_nav_state_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            images = batch["images"].to(self.device)
            instructions = batch["instruction"]
            actions = batch["actions"].to(self.device)
            trajectory_mask = batch["trajectory_mask"].to(self.device)
            success_flags = batch["success"].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.scaler:
                with autocast():
                    outputs = self.model(images, instructions, trajectory_mask)
                    loss_dict = self._compute_losses(outputs, actions, success_flags, trajectory_mask)
                    total_loss_batch = sum(loss_dict.values())
            else:
                outputs = self.model(images, instructions, trajectory_mask)
                loss_dict = self._compute_losses(outputs, actions, success_flags, trajectory_mask)
                total_loss_batch = sum(loss_dict.values())
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(total_loss_batch).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get("training", {}).get("max_grad_norm", 1.0)
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get("training", {}).get("max_grad_norm", 1.0)
                )
                self.optimizer.step()
            
            # Update metrics
            total_loss += total_loss_batch.item()
            total_action_loss += loss_dict["action_loss"].item()
            total_nav_state_loss += loss_dict["nav_state_loss"].item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{total_loss / num_batches:.4f}",
                "action_loss": f"{total_action_loss / num_batches:.4f}",
                "nav_loss": f"{total_nav_state_loss / num_batches:.4f}"
            })
            
            # Log batch metrics
            if batch_idx % 100 == 0:
                self._log_batch_metrics(loss_dict, batch_idx)
        
        # Compute epoch metrics
        epoch_metrics = {
            "train_loss": total_loss / num_batches,
            "train_action_loss": total_action_loss / num_batches,
            "train_nav_state_loss": total_nav_state_loss / num_batches
        }
        
        return epoch_metrics
    
    def _compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        success_flags: torch.Tensor,
        trajectory_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses.
        
        Args:
            outputs: Model outputs
            actions: Ground truth actions
            success_flags: Success flags
            trajectory_mask: Trajectory mask
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Action prediction loss
        action_logits = outputs["action_logits"]
        batch_size, seq_len, num_actions = action_logits.shape
        
        # Reshape for loss computation
        action_logits_flat = action_logits.view(-1, num_actions)
        actions_flat = actions.view(-1)
        
        # Apply mask to ignore padding
        mask = trajectory_mask.view(-1).bool()
        if mask.sum() > 0:
            losses["action_loss"] = self.criterion["action_loss"](
                action_logits_flat[mask], actions_flat[mask]
            )
        else:
            losses["action_loss"] = torch.tensor(0.0, device=self.device)
        
        # Navigation state loss
        nav_state_probs = outputs["nav_state_probs"]
        nav_state_targets = success_flags.unsqueeze(-1).expand_as(nav_state_probs)
        
        losses["nav_state_loss"] = self.criterion["nav_state_loss"](
            nav_state_probs.view(-1), nav_state_targets.view(-1)
        )
        
        # Planning loss (if available)
        if "planning_scores" in outputs:
            planning_scores = outputs["planning_scores"]
            # Simplified planning target (could be more sophisticated)
            planning_targets = torch.ones_like(planning_scores) * 0.5
            losses["planning_loss"] = self.criterion["planning_loss"](
                planning_scores.view(-1), planning_targets.view(-1)
            )
        else:
            losses["planning_loss"] = torch.tensor(0.0, device=self.device)
        
        return losses
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set.
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Compute VLN-specific metrics
        vln_metrics = evaluate_vln_model(
            self.model, self.val_dataloader, self.device
        )
        
        # Compute additional training metrics
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                images = batch["images"].to(self.device)
                instructions = batch["instruction"]
                actions = batch["actions"].to(self.device)
                trajectory_mask = batch["trajectory_mask"].to(self.device)
                success_flags = batch["success"].to(self.device)
                
                outputs = self.model(images, instructions, trajectory_mask)
                loss_dict = self._compute_losses(outputs, actions, success_flags, trajectory_mask)
                
                total_loss += sum(loss_dict.values()).item()
                num_batches += 1
        
        vln_metrics["val_loss"] = total_loss / num_batches
        
        return vln_metrics
    
    def _log_batch_metrics(self, loss_dict: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Log batch-level metrics.
        
        Args:
            loss_dict: Dictionary of losses
            batch_idx: Current batch index
        """
        if self.writer:
            for loss_name, loss_value in loss_dict.items():
                self.writer.add_scalar(
                    f"batch/{loss_name}", loss_value.item(),
                    self.current_epoch * len(self.train_dataloader) + batch_idx
                )
    
    def train(self) -> Dict[str, Any]:
        """Main training loop.
        
        Returns:
            Training results and best metrics
        """
        training_config = self.config.get("training", {})
        epochs = training_config.get("epochs", 100)
        save_every = training_config.get("save_every", 10)
        eval_every = training_config.get("eval_every", 5)
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Evaluation
            if epoch % eval_every == 0:
                val_metrics = self.evaluate()
                
                # Log metrics
                self._log_epoch_metrics(train_metrics, val_metrics, epoch)
                
                # Check for best model
                if self._is_best_model(val_metrics):
                    self.best_metrics = val_metrics.copy()
                    self._save_best_model(epoch, val_metrics)
                
                # Early stopping
                if self.early_stopping(val_metrics.get("val_loss", float('inf')), self.model):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Save checkpoint
            if epoch % save_every == 0:
                self._save_checkpoint(epoch, train_metrics, val_metrics if epoch % eval_every == 0 else {})
        
        logger.info("Training completed")
        return {
            "best_metrics": self.best_metrics,
            "training_history": self.training_history
        }
    
    def _log_epoch_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch: int
    ) -> None:
        """Log epoch-level metrics.
        
        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics
            epoch: Current epoch
        """
        # TensorBoard logging
        if self.writer:
            for metric_name, value in train_metrics.items():
                self.writer.add_scalar(f"train/{metric_name}", value, epoch)
            
            for metric_name, value in val_metrics.items():
                self.writer.add_scalar(f"val/{metric_name}", value, epoch)
        
        # WandB logging
        if self.use_wandb:
            log_dict = {f"train/{k}": v for k, v in train_metrics.items()}
            log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
            wandb.log(log_dict, step=epoch)
        
        # Console logging
        print(f"\nEpoch {epoch}")
        print_metrics(train_metrics, "Training Metrics")
        print_metrics(val_metrics, "Validation Metrics")
        
        # Store in history
        self.training_history.append({
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics
        })
    
    def _is_best_model(self, val_metrics: Dict[str, float]) -> bool:
        """Check if current model is the best.
        
        Args:
            val_metrics: Current validation metrics
            
        Returns:
            True if this is the best model so far
        """
        if not self.best_metrics:
            return True
        
        # Use success rate as primary metric
        current_success_rate = val_metrics.get("success_rate", 0.0)
        best_success_rate = self.best_metrics.get("success_rate", 0.0)
        
        return current_success_rate > best_success_rate
    
    def _save_best_model(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save the best model.
        
        Args:
            epoch: Current epoch
            metrics: Current metrics
        """
        checkpoint_dir = self.config.get("paths", {}).get("checkpoint_dir", "./checkpoints")
        best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=metrics.get("val_loss", 0.0),
            metrics=metrics,
            filepath=best_model_path
        )
        
        logger.info(f"Best model saved at epoch {epoch} with success rate {metrics.get('success_rate', 0.0):.2f}%")
    
    def _save_checkpoint(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ) -> None:
        """Save training checkpoint.
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        checkpoint_dir = self.config.get("paths", {}).get("checkpoint_dir", "./checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=train_metrics.get("train_loss", 0.0),
            metrics={**train_metrics, **val_metrics},
            filepath=checkpoint_path
        )
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.writer:
            self.writer.close()
        
        if self.use_wandb:
            wandb.finish()


def train_vln_model(
    config: Dict[str, Any],
    model: Optional[nn.Module] = None,
    resume_from_checkpoint: Optional[str] = None
) -> Dict[str, Any]:
    """Train a VLN model with given configuration.
    
    Args:
        config: Training configuration
        model: Model to train (optional, will create if not provided)
        resume_from_checkpoint: Path to checkpoint to resume from
        
    Returns:
        Training results
    """
    # Set random seed
    set_seed(config.get("seed", 42), config.get("deterministic", True))
    
    # Create model if not provided
    if model is None:
        model_config = config.get("model", {})
        model_name = model_config.get("name", "clip_vln")
        
        if model_name == "clip_vln":
            model = CLIPVLNModel(**model_config)
        elif model_name == "advanced_vln":
            model = AdvancedVLNModel(**model_config)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    # Create datasets
    data_config = config.get("data", {})
    train_dataset = VLNDataset(
        data_dir=data_config.get("data_dir", "./data"),
        split=data_config.get("train_split", "train"),
        **{k: v for k, v in data_config.items() if k not in ["data_dir", "train_split"]}
    )
    
    val_dataset = VLNDataset(
        data_dir=data_config.get("data_dir", "./data"),
        split=data_config.get("val_split", "val"),
        **{k: v for k, v in data_config.items() if k not in ["data_dir", "train_split", "val_split"]}
    )
    
    # Create data loaders
    train_dataloader = create_dataloader(train_dataset, **data_config)
    val_dataloader = create_dataloader(val_dataset, **data_config)
    
    # Create trainer
    trainer = VLNTrainer(model, train_dataloader, val_dataloader, config)
    
    # Resume from checkpoint if specified
    if resume_from_checkpoint:
        load_checkpoint(resume_from_checkpoint, model, trainer.optimizer)
    
    # Train model
    results = trainer.train()
    
    # Cleanup
    trainer.cleanup()
    
    return results
