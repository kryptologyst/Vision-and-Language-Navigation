#!/usr/bin/env python3
"""Training script for Vision-and-Language Navigation models."""

import argparse
import yaml
import torch
import logging
from pathlib import Path

from src.train import train_vln_model
from src.models import CLIPVLNModel, AdvancedVLNModel
from src.utils import set_seed, get_device


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train VLN model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["clip_vln", "advanced_vln"],
        default="clip_vln",
        help="Model type to train"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to train on (auto, cuda, mps, cpu)"
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
    
    # Override model type if specified
    if args.model:
        config["model"]["name"] = args.model
    
    # Override device if specified
    if args.device != "auto":
        config["device"] = args.device
    
    logger.info(f"Starting training with config: {args.config}")
    logger.info(f"Model type: {config['model']['name']}")
    logger.info(f"Device: {get_device(config.get('device', 'auto'))}")
    
    # Set random seed
    set_seed(config.get("seed", 42), config.get("deterministic", True))
    
    # Create model
    model_config = config.get("model", {})
    if config["model"]["name"] == "clip_vln":
        model = CLIPVLNModel(**model_config)
    elif config["model"]["name"] == "advanced_vln":
        model = AdvancedVLNModel(**model_config)
    else:
        raise ValueError(f"Unknown model: {config['model']['name']}")
    
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train model
    try:
        results = train_vln_model(config, model, args.resume)
        
        logger.info("Training completed successfully!")
        logger.info(f"Best Success Rate: {results['best_metrics'].get('success_rate', 0.0):.2f}%")
        logger.info(f"Best SPL: {results['best_metrics'].get('spl', 0.0):.2f}%")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
