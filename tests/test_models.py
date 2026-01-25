"""Unit tests for Vision-and-Language Navigation models."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.models import CLIPVLNModel, AdvancedVLNModel, CrossAttentionLayer, PositionalEncoding
from src.data import VLNDataset, collate_fn
from src.eval import VLNMetrics, TrajectoryAnalyzer
from src.utils import get_device, set_seed, EarlyStopping


class TestPositionalEncoding:
    """Test positional encoding module."""
    
    def test_positional_encoding_shape(self):
        """Test positional encoding output shape."""
        d_model = 512
        max_len = 100
        pe = PositionalEncoding(d_model, max_len)
        
        x = torch.randn(50, 2, d_model)  # seq_len, batch_size, d_model
        output = pe(x)
        
        assert output.shape == x.shape
        assert not torch.equal(output, x)  # Should be different due to encoding
    
    def test_positional_encoding_deterministic(self):
        """Test that positional encoding is deterministic."""
        d_model = 256
        pe = PositionalEncoding(d_model)
        
        x = torch.randn(10, 1, d_model)
        output1 = pe(x)
        output2 = pe(x)
        
        assert torch.equal(output1, output2)


class TestCrossAttentionLayer:
    """Test cross-attention layer."""
    
    def test_cross_attention_shape(self):
        """Test cross-attention output shape."""
        d_model = 512
        num_heads = 8
        cross_attn = CrossAttentionLayer(d_model, num_heads)
        
        batch_size = 2
        seq_len_q = 10
        seq_len_kv = 15
        
        query = torch.randn(batch_size, seq_len_q, d_model)
        key = torch.randn(batch_size, seq_len_kv, d_model)
        value = torch.randn(batch_size, seq_len_kv, d_model)
        
        output = cross_attn(query, key, value)
        
        assert output.shape == (batch_size, seq_len_q, d_model)
    
    def test_cross_attention_with_mask(self):
        """Test cross-attention with mask."""
        d_model = 256
        num_heads = 4
        cross_attn = CrossAttentionLayer(d_model, num_heads)
        
        batch_size = 1
        seq_len_q = 5
        seq_len_kv = 8
        
        query = torch.randn(batch_size, seq_len_q, d_model)
        key = torch.randn(batch_size, seq_len_kv, d_model)
        value = torch.randn(batch_size, seq_len_kv, d_model)
        
        # Create mask (1 for valid positions, 0 for masked)
        mask = torch.ones(batch_size, seq_len_kv)
        mask[:, -2:] = 0  # Mask last 2 positions
        
        output = cross_attn(query, key, value, mask)
        
        assert output.shape == (batch_size, seq_len_q, d_model)


class TestCLIPVLNModel:
    """Test CLIP VLN model."""
    
    @pytest.fixture
    def model(self):
        """Create CLIP VLN model for testing."""
        return CLIPVLNModel(
            hidden_size=256,
            num_layers=2,
            num_heads=4,
            max_trajectory_length=10
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        batch_size = 2
        seq_len = 5
        channels = 3
        height = 224
        width = 224
        
        images = torch.randn(batch_size, seq_len, channels, height, width)
        instructions = ["Go straight", "Turn left"]
        trajectory_mask = torch.ones(batch_size, seq_len)
        
        return images, instructions, trajectory_mask
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert isinstance(model, CLIPVLNModel)
        assert model.hidden_size == 256
        assert model.max_trajectory_length == 10
    
    @patch('src.models.CLIPModel.from_pretrained')
    @patch('src.models.CLIPProcessor.from_pretrained')
    def test_forward_pass(self, mock_processor, mock_model, model, sample_data):
        """Test model forward pass."""
        images, instructions, trajectory_mask = sample_data
        
        # Mock CLIP model and processor
        mock_clip_model = Mock()
        mock_clip_model.get_text_features.return_value = torch.randn(2, 512)
        mock_clip_model.get_image_features.return_value = torch.randn(10, 512)
        mock_model.return_value = mock_clip_model
        
        mock_clip_processor = Mock()
        mock_clip_processor.return_value = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }
        mock_processor.return_value = mock_clip_processor
        
        model.clip_model = mock_clip_model
        model.clip_processor = mock_clip_processor
        
        # Forward pass
        outputs = model(images, instructions, trajectory_mask)
        
        # Check output keys
        expected_keys = ["action_logits", "nav_state_probs", "trajectory_features"]
        for key in expected_keys:
            assert key in outputs
        
        # Check output shapes
        assert outputs["action_logits"].shape == (2, 5, 4)  # batch_size, seq_len, num_actions
        assert outputs["nav_state_probs"].shape == (2, 5, 1)  # batch_size, seq_len, 1
        assert outputs["trajectory_features"].shape == (2, 5, 256)  # batch_size, seq_len, hidden_size
    
    def test_encode_instruction(self, model):
        """Test instruction encoding."""
        instructions = ["Go straight", "Turn left"]
        
        with patch.object(model.clip_processor, '__call__') as mock_call:
            mock_call.return_value = {
                'input_ids': torch.randint(0, 100, (2, 10)),
                'attention_mask': torch.ones(2, 10)
            }
            
            with patch.object(model.clip_model, 'get_text_features') as mock_text_features:
                mock_text_features.return_value = torch.randn(2, 512)
                
                features = model.encode_instruction(instructions)
                
                assert features.shape == (2, 256)  # batch_size, hidden_size
    
    def test_encode_visual_observations(self, model):
        """Test visual observation encoding."""
        batch_size = 2
        seq_len = 3
        images = torch.randn(batch_size, seq_len, 3, 224, 224)
        
        with patch.object(model.clip_model, 'get_image_features') as mock_image_features:
            mock_image_features.return_value = torch.randn(6, 512)  # batch_size * seq_len, clip_dim
            
            features = model.encode_visual_observations(images)
            
            assert features.shape == (batch_size, seq_len, 256)  # batch_size, seq_len, hidden_size


class TestAdvancedVLNModel:
    """Test Advanced VLN model."""
    
    @pytest.fixture
    def model(self):
        """Create Advanced VLN model for testing."""
        return AdvancedVLNModel(
            hidden_size=256,
            memory_size=128,
            num_layers=2,
            num_heads=4,
            max_trajectory_length=10
        )
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert isinstance(model, AdvancedVLNModel)
        assert model.memory_size == 128
        assert hasattr(model, 'memory_bank')
        assert hasattr(model, 'planning_head')
        assert hasattr(model, 'goal_head')
    
    def test_memory_bank_shape(self, model):
        """Test memory bank shape."""
        assert model.memory_bank.shape == (128, 256)  # memory_size, hidden_size


class TestVLNDataset:
    """Test VLN dataset."""
    
    @pytest.fixture
    def dataset(self):
        """Create VLN dataset for testing."""
        return VLNDataset(
            data_dir="./data",
            split="train",
            max_trajectory_length=10
        )
    
    def test_dataset_initialization(self, dataset):
        """Test dataset initialization."""
        assert isinstance(dataset, VLNDataset)
        assert dataset.split == "train"
        assert dataset.max_trajectory_length == 10
    
    def test_synthetic_data_generation(self, dataset):
        """Test synthetic data generation."""
        # This should generate synthetic data since no real data exists
        assert len(dataset.data) > 0
        
        # Check data format
        sample = dataset.data[0]
        required_keys = ["instruction", "trajectory", "goal_position", "success", "path_length"]
        for key in required_keys:
            assert key in sample
    
    def test_getitem(self, dataset):
        """Test dataset __getitem__ method."""
        if len(dataset) > 0:
            sample = dataset[0]
            
            required_keys = ["images", "instruction", "actions", "positions", "orientations", 
                           "trajectory_mask", "success", "path_length", "goal_position"]
            for key in required_keys:
                assert key in sample
            
            # Check tensor shapes
            assert sample["images"].shape == (10, 3, 224, 224)  # max_trajectory_length, channels, height, width
            assert sample["actions"].shape == (10,)  # max_trajectory_length
            assert sample["trajectory_mask"].shape == (10,)  # max_trajectory_length


class TestCollateFunction:
    """Test collate function."""
    
    def test_collate_fn(self):
        """Test collate function."""
        # Create sample batch
        batch = []
        for i in range(3):
            sample = {
                "images": torch.randn(10, 3, 224, 224),
                "instruction": f"instruction_{i}",
                "actions": torch.randint(0, 4, (10,)),
                "positions": torch.randn(10, 2),
                "orientations": torch.randn(10),
                "trajectory_mask": torch.ones(10),
                "success": torch.tensor(1.0),
                "path_length": torch.tensor(10),
                "goal_position": torch.randn(2)
            }
            batch.append(sample)
        
        # Apply collate function
        batched = collate_fn(batch)
        
        # Check batch shapes
        assert batched["images"].shape == (3, 10, 3, 224, 224)  # batch_size, seq_len, channels, height, width
        assert batched["actions"].shape == (3, 10)  # batch_size, seq_len
        assert len(batched["instruction"]) == 3  # batch_size
        
        # Check instruction list
        assert isinstance(batched["instruction"], list)
        assert batched["instruction"][0] == "instruction_0"


class TestVLNMetrics:
    """Test VLN metrics."""
    
    @pytest.fixture
    def metrics(self):
        """Create VLN metrics calculator."""
        return VLNMetrics()
    
    def test_metrics_initialization(self, metrics):
        """Test metrics initialization."""
        assert isinstance(metrics, VLNMetrics)
        assert len(metrics.predictions) == 0
        assert len(metrics.ground_truths) == 0
    
    def test_add_batch(self, metrics):
        """Test adding batch to metrics."""
        batch_size = 2
        seq_len = 5
        num_actions = 4
        
        predictions = torch.randn(batch_size, seq_len, num_actions)
        ground_truth = torch.randint(0, num_actions, (batch_size, seq_len))
        path_lengths = torch.tensor([5, 4])
        success_flags = torch.tensor([1.0, 0.0])
        
        metrics.add_batch(predictions, ground_truth, path_lengths, success_flags)
        
        assert len(metrics.predictions) == batch_size
        assert len(metrics.ground_truths) == batch_size
        assert len(metrics.path_lengths) == batch_size
        assert len(metrics.success_flags) == batch_size
    
    def test_compute_success_rate(self, metrics):
        """Test success rate computation."""
        # Add some test data
        metrics.success_flags = [1.0, 0.0, 1.0, 1.0, 0.0]
        
        success_rate = metrics.compute_success_rate()
        
        assert success_rate == 60.0  # 3 out of 5 successful
    
    def test_compute_spl(self, metrics):
        """Test SPL computation."""
        # Add test data
        metrics.success_flags = [1.0, 0.0, 1.0]
        metrics.path_lengths = [5, 3, 4]
        
        spl = metrics.compute_spl()
        
        assert isinstance(spl, float)
        assert 0.0 <= spl <= 100.0
    
    def test_reset(self, metrics):
        """Test metrics reset."""
        # Add some data
        metrics.predictions = [torch.randn(5, 4)]
        metrics.ground_truths = [torch.randint(0, 4, (5,))]
        
        # Reset
        metrics.reset()
        
        assert len(metrics.predictions) == 0
        assert len(metrics.ground_truths) == 0


class TestTrajectoryAnalyzer:
    """Test trajectory analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create trajectory analyzer."""
        return TrajectoryAnalyzer()
    
    def test_trajectory_similarity(self, analyzer):
        """Test trajectory similarity computation."""
        traj1 = [(0, 0), (1, 1), (2, 2)]
        traj2 = [(0, 0), (1, 1), (2, 2)]
        
        similarity = analyzer.compute_trajectory_similarity(traj1, traj2)
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
    
    def test_path_efficiency(self, analyzer):
        """Test path efficiency computation."""
        actual_path = [(0, 0), (1, 1), (2, 2), (3, 3)]
        optimal_path = [(0, 0), (3, 3)]
        
        efficiency = analyzer.compute_path_efficiency(actual_path, optimal_path)
        
        assert isinstance(efficiency, float)
        assert 0.0 <= efficiency <= 1.0
    
    def test_path_length_computation(self, analyzer):
        """Test path length computation."""
        path = [(0, 0), (3, 4), (6, 8)]
        
        length = analyzer._compute_path_length(path)
        
        # Should be 5 + 5 = 10
        assert abs(length - 10.0) < 1e-6


class TestUtils:
    """Test utility functions."""
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # This is hard to test directly, but we can check it doesn't raise an error
        assert True
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        early_stopping = EarlyStopping(patience=3)
        model = Mock()
        
        # Test with improving loss
        assert not early_stopping(0.5, model)
        assert not early_stopping(0.4, model)
        assert not early_stopping(0.3, model)
        
        # Test with non-improving loss
        assert not early_stopping(0.3, model)  # patience = 1
        assert not early_stopping(0.3, model)  # patience = 2
        assert early_stopping(0.3, model)  # patience = 3, should stop


if __name__ == "__main__":
    pytest.main([__file__])
