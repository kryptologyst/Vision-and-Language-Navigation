#!/usr/bin/env python3
"""Evaluation script for Vision-and-Language Navigation models."""

import argparse
import yaml
import torch
import logging
from pathlib import Path

from src.models import CLIPVLNModel, AdvancedVLNModel
from src.data import VLNDataset, create_dataloader
from src.eval import evaluate_vln_model, print_metrics
from src.utils import get_device, load_checkpoint


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('evaluation.log')
        ]
    )


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate VLN model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to evaluate on (auto, cuda, mps, cpu)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Generate detailed evaluation report"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save model predictions"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override device if specified
    if args.device != "auto":
        config["device"] = args.device
    
    device = get_device(config.get("device", "auto"))
    logger.info(f"Evaluating on device: {device}")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    
    # Create model
    model_config = checkpoint.get("model_config", config.get("model", {}))
    model_name = model_config.get("name", "clip_vln")
    
    if model_name == "clip_vln":
        model = CLIPVLNModel(**model_config)
    elif model_name == "advanced_vln":
        model = AdvancedVLNModel(**model_config)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model: {model_name}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset and dataloader
    data_config = config.get("data", {})
    dataset = VLNDataset(
        data_dir=data_config.get("data_dir", "./data"),
        split=args.split,
        **{k: v for k, v in data_config.items() if k not in ["data_dir"]}
    )
    
    dataloader = create_dataloader(
        dataset,
        batch_size=data_config.get("batch_size", 16),
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True)
    )
    
    logger.info(f"Evaluating on {args.split} split: {len(dataset)} samples")
    
    # Evaluate model
    try:
        metrics = evaluate_vln_model(model, dataloader, device)
        
        # Print results
        print_metrics(metrics, f"Evaluation Results ({args.split})")
        
        # Save predictions if requested
        if args.save_predictions:
            predictions_path = f"predictions_{args.split}.json"
            logger.info(f"Saving predictions to {predictions_path}")
            # TODO: Implement prediction saving
        
        # Generate detailed report if requested
        if args.detailed:
            logger.info("Generating detailed evaluation report...")
            # TODO: Implement detailed reporting
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
