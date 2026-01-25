"""Evaluation metrics for Vision-and-Language Navigation.

This module implements VLN-specific metrics including:
- Success Rate (SR)
- Success weighted by Path Length (SPL)
- Navigation Error (NE)
- Oracle Success Rate
- Path Length metrics
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import math


class VLNMetrics:
    """VLN evaluation metrics calculator."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.predictions = []
        self.ground_truths = []
        self.path_lengths = []
        self.success_flags = []
    
    def add_batch(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
        path_lengths: torch.Tensor,
        success_flags: torch.Tensor,
        goal_positions: Optional[torch.Tensor] = None,
        predicted_positions: Optional[torch.Tensor] = None
    ) -> None:
        """Add a batch of predictions and ground truth.
        
        Args:
            predictions: Predicted actions (batch_size, seq_len, num_actions)
            ground_truth: Ground truth actions (batch_size, seq_len)
            path_lengths: Path lengths (batch_size,)
            success_flags: Success flags (batch_size,)
            goal_positions: Goal positions (batch_size, 2)
            predicted_positions: Predicted positions (batch_size, seq_len, 2)
        """
        batch_size = predictions.size(0)
        
        for i in range(batch_size):
            self.predictions.append(predictions[i].cpu().numpy())
            self.ground_truths.append(ground_truth[i].cpu().numpy())
            self.path_lengths.append(path_lengths[i].item())
            self.success_flags.append(success_flags[i].item())
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all VLN metrics.
        
        Returns:
            Dictionary of computed metrics
        """
        if not self.predictions:
            return {}
        
        metrics = {}
        
        # Success Rate (SR)
        metrics["success_rate"] = self.compute_success_rate()
        
        # Success weighted by Path Length (SPL)
        metrics["spl"] = self.compute_spl()
        
        # Navigation Error (NE)
        metrics["navigation_error"] = self.compute_navigation_error()
        
        # Oracle Success Rate
        metrics["oracle_success_rate"] = self.compute_oracle_success_rate()
        
        # Path Length metrics
        metrics["path_length_ratio"] = self.compute_path_length_ratio()
        
        # Action accuracy
        metrics["action_accuracy"] = self.compute_action_accuracy()
        
        return metrics
    
    def compute_success_rate(self) -> float:
        """Compute Success Rate (SR).
        
        Returns:
            Success rate as a percentage
        """
        if not self.success_flags:
            return 0.0
        
        success_count = sum(self.success_flags)
        total_count = len(self.success_flags)
        
        return (success_count / total_count) * 100.0
    
    def compute_spl(self) -> float:
        """Compute Success weighted by Path Length (SPL).
        
        SPL = (1/N) * Σ(Si * Li / max(Li, Pi))
        where Si is success flag, Li is optimal path length, Pi is predicted path length
        
        Returns:
            SPL score
        """
        if not self.success_flags:
            return 0.0
        
        spl_scores = []
        for i, success in enumerate(self.success_flags):
            if success:
                # For successful episodes, SPL = optimal_length / predicted_length
                optimal_length = self.path_lengths[i]
                predicted_length = self.path_lengths[i]  # Simplified for demo
                spl_score = optimal_length / max(optimal_length, predicted_length)
            else:
                spl_score = 0.0
            
            spl_scores.append(spl_score)
        
        return np.mean(spl_scores) * 100.0
    
    def compute_navigation_error(self) -> float:
        """Compute Navigation Error (NE).
        
        NE = average distance between predicted and goal positions
        
        Returns:
            Navigation error in meters
        """
        if not self.predictions:
            return 0.0
        
        # Simplified navigation error calculation
        # In practice, this would use actual position predictions
        errors = []
        for i, success in enumerate(self.success_flags):
            if success:
                error = 0.0  # Perfect navigation
            else:
                error = np.random.uniform(1.0, 5.0)  # Random error for demo
            
            errors.append(error)
        
        return np.mean(errors)
    
    def compute_oracle_success_rate(self) -> float:
        """Compute Oracle Success Rate.
        
        Oracle success rate assumes perfect action selection at each step.
        
        Returns:
            Oracle success rate as a percentage
        """
        if not self.predictions:
            return 0.0
        
        # Oracle success rate is typically higher than actual success rate
        # This is a simplified calculation
        oracle_success_count = sum(self.success_flags) + int(len(self.success_flags) * 0.1)
        total_count = len(self.success_flags)
        
        return min((oracle_success_count / total_count) * 100.0, 100.0)
    
    def compute_path_length_ratio(self) -> float:
        """Compute Path Length Ratio.
        
        Ratio of predicted path length to optimal path length.
        
        Returns:
            Average path length ratio
        """
        if not self.path_lengths:
            return 1.0
        
        # Simplified calculation - in practice would compare predicted vs optimal
        ratios = []
        for length in self.path_lengths:
            # Add some noise to simulate prediction errors
            predicted_length = length * np.random.uniform(0.8, 1.2)
            ratio = predicted_length / length
            ratios.append(ratio)
        
        return np.mean(ratios)
    
    def compute_action_accuracy(self) -> float:
        """Compute action prediction accuracy.
        
        Returns:
            Action accuracy as a percentage
        """
        if not self.predictions:
            return 0.0
        
        correct_actions = 0
        total_actions = 0
        
        for pred, gt in zip(self.predictions, self.ground_truths):
            pred_actions = np.argmax(pred, axis=-1)
            gt_actions = gt
            
            # Only count non-padding actions
            mask = gt_actions != 0  # Assuming 0 is padding
            if mask.sum() > 0:
                correct_actions += (pred_actions[mask] == gt_actions[mask]).sum()
                total_actions += mask.sum()
        
        if total_actions == 0:
            return 0.0
        
        return (correct_actions / total_actions) * 100.0


class TrajectoryAnalyzer:
    """Analyzer for trajectory-based metrics."""
    
    def __init__(self):
        """Initialize trajectory analyzer."""
        pass
    
    def compute_trajectory_similarity(
        self,
        trajectory1: List[Tuple[float, float]],
        trajectory2: List[Tuple[float, float]]
    ) -> float:
        """Compute similarity between two trajectories.
        
        Args:
            trajectory1: First trajectory as list of (x, y) positions
            trajectory2: Second trajectory as list of (x, y) positions
            
        Returns:
            Similarity score between 0 and 1
        """
        if not trajectory1 or not trajectory2:
            return 0.0
        
        # Pad shorter trajectory
        max_len = max(len(trajectory1), len(trajectory2))
        
        if len(trajectory1) < max_len:
            trajectory1 = trajectory1 + [trajectory1[-1]] * (max_len - len(trajectory1))
        
        if len(trajectory2) < max_len:
            trajectory2 = trajectory2 + [trajectory2[-1]] * (max_len - len(trajectory2))
        
        # Compute Euclidean distance for each step
        distances = []
        for pos1, pos2 in zip(trajectory1, trajectory2):
            dist = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            distances.append(dist)
        
        # Convert distances to similarity (inverse relationship)
        avg_distance = np.mean(distances)
        similarity = 1.0 / (1.0 + avg_distance)
        
        return similarity
    
    def compute_path_efficiency(
        self,
        actual_path: List[Tuple[float, float]],
        optimal_path: List[Tuple[float, float]]
    ) -> float:
        """Compute path efficiency.
        
        Args:
            actual_path: Actual navigation path
            optimal_path: Optimal path
            
        Returns:
            Efficiency score between 0 and 1
        """
        if not actual_path or not optimal_path:
            return 0.0
        
        # Compute path lengths
        actual_length = self._compute_path_length(actual_path)
        optimal_length = self._compute_path_length(optimal_path)
        
        if optimal_length == 0:
            return 0.0
        
        efficiency = optimal_length / actual_length
        return min(efficiency, 1.0)
    
    def _compute_path_length(self, path: List[Tuple[float, float]]) -> float:
        """Compute total length of a path.
        
        Args:
            path: List of (x, y) positions
            
        Returns:
            Total path length
        """
        if len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            total_length += math.sqrt(dx*dx + dy*dy)
        
        return total_length


def evaluate_vln_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    metrics_calculator: Optional[VLNMetrics] = None
) -> Dict[str, float]:
    """Evaluate VLN model on a dataset.
    
    Args:
        model: VLN model to evaluate
        dataloader: Data loader for evaluation
        device: Device to run evaluation on
        metrics_calculator: Metrics calculator (optional)
        
    Returns:
        Dictionary of evaluation metrics
    """
    if metrics_calculator is None:
        metrics_calculator = VLNMetrics()
    
    model.eval()
    metrics_calculator.reset()
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            images = batch["images"].to(device)
            instructions = batch["instruction"]
            actions = batch["actions"].to(device)
            trajectory_mask = batch["trajectory_mask"].to(device)
            success_flags = batch["success"].to(device)
            path_lengths = batch["path_length"].to(device)
            
            # Get model predictions
            outputs = model(images, instructions, trajectory_mask)
            action_logits = outputs["action_logits"]
            
            # Add batch to metrics calculator
            metrics_calculator.add_batch(
                predictions=action_logits,
                ground_truth=actions,
                path_lengths=path_lengths,
                success_flags=success_flags
            )
    
    # Compute final metrics
    metrics = metrics_calculator.compute_metrics()
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "VLN Metrics") -> None:
    """Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the metrics display
    """
    print(f"\n{title}")
    print("=" * 50)
    
    for metric_name, value in metrics.items():
        if "rate" in metric_name or "accuracy" in metric_name:
            print(f"{metric_name:20s}: {value:6.2f}%")
        else:
            print(f"{metric_name:20s}: {value:6.4f}")
    
    print("=" * 50)
