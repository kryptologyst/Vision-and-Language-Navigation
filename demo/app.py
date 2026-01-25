"""Interactive demo application for Vision-and-Language Navigation.

This module provides both Streamlit and Gradio interfaces for VLN demonstration.
"""

import streamlit as st
import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Any
import io
import base64

from ..models import CLIPVLNModel, AdvancedVLNModel
from ..utils import get_device, load_checkpoint
from ..eval import VLNMetrics, TrajectoryAnalyzer


class VLNDemo:
    """VLN demonstration application."""
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """Initialize VLN demo.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.device = get_device(device)
        self.model = None
        self.metrics_calculator = VLNMetrics()
        self.trajectory_analyzer = TrajectoryAnalyzer()
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
        else:
            self.load_default_model()
    
    def load_model(self, model_path: str) -> None:
        """Load trained model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
        """
        try:
            checkpoint = load_checkpoint(model_path, device=self.device)
            
            # Create model based on checkpoint
            model_config = checkpoint.get("model_config", {})
            model_name = model_config.get("name", "clip_vln")
            
            if model_name == "clip_vln":
                self.model = CLIPVLNModel(**model_config)
            elif model_name == "advanced_vln":
                self.model = AdvancedVLNModel(**model_config)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            self.model.to(self.device)
            self.model.eval()
            
            st.success(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            self.load_default_model()
    
    def load_default_model(self) -> None:
        """Load default model for demonstration."""
        try:
            self.model = CLIPVLNModel()
            self.model.to(self.device)
            self.model.eval()
            st.info("Using default CLIP VLN model for demonstration")
        except Exception as e:
            st.error(f"Failed to load default model: {str(e)}")
    
    def predict_navigation(
        self,
        images: List[Image.Image],
        instruction: str,
        show_attention: bool = True
    ) -> Dict[str, Any]:
        """Predict navigation actions for given images and instruction.
        
        Args:
            images: List of visual observations
            instruction: Navigation instruction
            show_attention: Whether to compute attention maps
            
        Returns:
            Dictionary containing predictions and visualizations
        """
        if not self.model:
            return {"error": "No model loaded"}
        
        try:
            # Preprocess images
            processed_images = []
            for img in images:
                # Resize and convert to tensor
                img = img.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
                processed_images.append(img_tensor)
            
            # Stack images
            images_tensor = torch.stack(processed_images).unsqueeze(0).to(self.device)
            
            # Create trajectory mask
            trajectory_mask = torch.ones(1, len(images)).to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model([instruction], images_tensor, trajectory_mask)
            
            # Extract predictions
            action_logits = outputs["action_logits"][0]  # Remove batch dimension
            nav_state_probs = outputs["nav_state_probs"][0]
            
            # Get predicted actions
            predicted_actions = torch.argmax(action_logits, dim=-1).cpu().numpy()
            
            # Action names
            action_names = ["Forward", "Left", "Right", "Stop"]
            predicted_action_names = [action_names[action] for action in predicted_actions]
            
            # Compute attention maps if requested
            attention_maps = None
            if show_attention and "trajectory_features" in outputs:
                attention_maps = self._compute_attention_maps(outputs)
            
            return {
                "predicted_actions": predicted_action_names,
                "action_probabilities": torch.softmax(action_logits, dim=-1).cpu().numpy(),
                "nav_state_probabilities": nav_state_probs.cpu().numpy(),
                "attention_maps": attention_maps,
                "instruction_features": outputs.get("instruction_features", None),
                "visual_features": outputs.get("visual_features", None)
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def _compute_attention_maps(self, outputs: Dict[str, torch.Tensor]) -> np.ndarray:
        """Compute attention maps from model outputs.
        
        Args:
            outputs: Model outputs
            
        Returns:
            Attention maps
        """
        try:
            # This is a simplified attention computation
            # In practice, you would extract attention weights from the model
            trajectory_features = outputs["trajectory_features"][0]  # Remove batch dimension
            
            # Compute attention as feature magnitude
            attention_scores = torch.norm(trajectory_features, dim=-1).cpu().numpy()
            
            # Normalize attention scores
            attention_scores = attention_scores / np.max(attention_scores)
            
            return attention_scores
            
        except Exception as e:
            return np.zeros(10)  # Return dummy attention
    
    def visualize_trajectory(
        self,
        images: List[Image.Image],
        predicted_actions: List[str],
        attention_scores: Optional[np.ndarray] = None
    ) -> plt.Figure:
        """Visualize navigation trajectory.
        
        Args:
            images: Visual observations
            predicted_actions: Predicted actions
            attention_scores: Attention scores for each step
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for i, (img, action) in enumerate(zip(images, predicted_actions)):
            if i >= len(axes):
                break
            
            ax = axes[i]
            ax.imshow(img)
            ax.set_title(f"Step {i+1}: {action}", fontsize=10)
            ax.axis('off')
            
            # Add attention visualization if available
            if attention_scores is not None and i < len(attention_scores):
                attention_score = attention_scores[i]
                # Add color overlay based on attention
                overlay = np.zeros((*img.size[::-1], 4))
                overlay[:, :, 3] = attention_score * 0.3  # Alpha based on attention
                overlay[:, :, 0] = attention_score  # Red channel
                ax.imshow(overlay)
        
        # Hide unused subplots
        for i in range(len(images), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_action_probability_plot(self, action_probs: np.ndarray) -> plt.Figure:
        """Create action probability visualization.
        
        Args:
            action_probs: Action probabilities (seq_len, num_actions)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        action_names = ["Forward", "Left", "Right", "Stop"]
        steps = range(len(action_probs))
        
        # Create stacked bar chart
        bottom = np.zeros(len(steps))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, action_name in enumerate(action_names):
            ax.bar(steps, action_probs[:, i], bottom=bottom, 
                  label=action_name, color=colors[i], alpha=0.7)
            bottom += action_probs[:, i]
        
        ax.set_xlabel('Navigation Step')
        ax.set_ylabel('Action Probability')
        ax.set_title('Action Probability Distribution Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def create_attention_heatmap(self, attention_scores: np.ndarray) -> plt.Figure:
        """Create attention heatmap visualization.
        
        Args:
            attention_scores: Attention scores for each step
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Create heatmap
        attention_matrix = attention_scores.reshape(1, -1)
        sns.heatmap(attention_matrix, cmap='YlOrRd', cbar=True, ax=ax)
        
        ax.set_xlabel('Navigation Step')
        ax.set_ylabel('Attention')
        ax.set_title('Attention Scores Over Navigation Steps')
        
        return fig


def create_streamlit_app():
    """Create Streamlit demo application."""
    st.set_page_config(
        page_title="VLN Navigation Demo",
        page_icon="🧭",
        layout="wide"
    )
    
    st.title("🧭 Vision-and-Language Navigation Demo")
    st.markdown("Interactive demonstration of VLN models for navigation tasks")
    
    # Initialize demo
    if 'demo' not in st.session_state:
        st.session_state.demo = VLNDemo()
    
    demo = st.session_state.demo
    
    # Sidebar for model selection
    with st.sidebar:
        st.header("Model Configuration")
        
        model_path = st.text_input(
            "Model Checkpoint Path",
            value="",
            help="Path to trained model checkpoint (optional)"
        )
        
        if st.button("Load Model") and model_path:
            demo.load_model(model_path)
        
        st.markdown("---")
        st.markdown("### Sample Instructions")
        sample_instructions = [
            "Go straight and turn left at the kitchen",
            "Walk to the bedroom and stop",
            "Turn right and go to the living room",
            "Navigate to the bathroom door",
            "Go forward and turn left at the hallway"
        ]
        
        selected_instruction = st.selectbox(
            "Choose a sample instruction:",
            sample_instructions
        )
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input")
        
        # Instruction input
        instruction = st.text_area(
            "Navigation Instruction",
            value=selected_instruction,
            height=100,
            help="Describe the navigation task"
        )
        
        # Image upload
        st.subheader("Visual Observations")
        uploaded_images = st.file_uploader(
            "Upload images (in order of observation)",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload images representing the visual observations during navigation"
        )
        
        # Generate synthetic images if none uploaded
        if not uploaded_images:
            st.info("No images uploaded. Using synthetic images for demonstration.")
            synthetic_images = []
            for i in range(5):
                # Create synthetic image
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                synthetic_img = Image.fromarray(img_array)
                synthetic_images.append(synthetic_img)
            uploaded_images = synthetic_images
        
        # Display uploaded images
        if uploaded_images:
            st.subheader("Uploaded Images")
            cols = st.columns(min(len(uploaded_images), 5))
            for i, img in enumerate(uploaded_images):
                with cols[i % 5]:
                    st.image(img, caption=f"Step {i+1}", use_column_width=True)
    
    with col2:
        st.header("Navigation Prediction")
        
        # Prediction options
        show_attention = st.checkbox("Show Attention Maps", value=True)
        
        if st.button("Predict Navigation", type="primary"):
            if instruction and uploaded_images:
                with st.spinner("Computing navigation predictions..."):
                    # Convert uploaded files to PIL Images
                    images = []
                    for uploaded_file in uploaded_images:
                        image = Image.open(uploaded_file)
                        images.append(image)
                    
                    # Get predictions
                    results = demo.predict_navigation(
                        images, instruction, show_attention
                    )
                    
                    if "error" in results:
                        st.error(results["error"])
                    else:
                        # Display predictions
                        st.subheader("Predicted Actions")
                        for i, action in enumerate(results["predicted_actions"]):
                            st.write(f"Step {i+1}: **{action}**")
                        
                        # Action probabilities
                        if "action_probabilities" in results:
                            st.subheader("Action Probabilities")
                            action_probs = results["action_probabilities"]
                            
                            # Create probability plot
                            prob_fig = demo.create_action_probability_plot(action_probs)
                            st.pyplot(prob_fig)
                        
                        # Attention visualization
                        if show_attention and "attention_maps" in results:
                            st.subheader("Attention Visualization")
                            attention_scores = results["attention_maps"]
                            
                            # Create attention heatmap
                            attention_fig = demo.create_attention_heatmap(attention_scores)
                            st.pyplot(attention_fig)
                            
                            # Create trajectory visualization
                            trajectory_fig = demo.visualize_trajectory(
                                images, results["predicted_actions"], attention_scores
                            )
                            st.pyplot(trajectory_fig)
                        
                        # Navigation state probabilities
                        if "nav_state_probabilities" in results:
                            st.subheader("Navigation State")
                            nav_probs = results["nav_state_probabilities"]
                            
                            for i, prob in enumerate(nav_probs):
                                st.write(f"Step {i+1}: {prob[0]:.2%} confidence in navigation state")
            else:
                st.warning("Please provide both instruction and images")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Vision-and-Language Navigation Demo** | "
        "Built with Streamlit and PyTorch | "
        "Advanced Computer Vision Project"
    )


def create_gradio_app():
    """Create Gradio demo application."""
    
    def gradio_predict(instruction, *images):
        """Gradio prediction function."""
        # Filter out None images
        valid_images = [img for img in images if img is not None]
        
        if not valid_images or not instruction:
            return "Please provide both instruction and at least one image"
        
        # Initialize demo
        demo = VLNDemo()
        
        # Get predictions
        results = demo.predict_navigation(valid_images, instruction)
        
        if "error" in results:
            return results["error"]
        
        # Format results
        output = f"**Instruction:** {instruction}\n\n"
        output += "**Predicted Actions:**\n"
        
        for i, action in enumerate(results["predicted_actions"]):
            output += f"Step {i+1}: {action}\n"
        
        return output
    
    # Create Gradio interface
    with gr.Blocks(title="VLN Navigation Demo") as app:
        gr.Markdown("# 🧭 Vision-and-Language Navigation Demo")
        
        with gr.Row():
            with gr.Column():
                instruction_input = gr.Textbox(
                    label="Navigation Instruction",
                    placeholder="Enter navigation instruction...",
                    lines=3
                )
                
                image_inputs = []
                for i in range(5):
                    image_input = gr.Image(
                        label=f"Visual Observation {i+1}",
                        type="pil"
                    )
                    image_inputs.append(image_input)
                
                predict_btn = gr.Button("Predict Navigation", variant="primary")
            
            with gr.Column():
                output_text = gr.Markdown(label="Navigation Prediction")
        
        # Set up event handlers
        predict_btn.click(
            fn=gradio_predict,
            inputs=[instruction_input] + image_inputs,
            outputs=output_text
        )
        
        # Add examples
        gr.Examples(
            examples=[
                ["Go straight and turn left at the kitchen"] + [None] * 5,
                ["Walk to the bedroom and stop"] + [None] * 5,
                ["Turn right and go to the living room"] + [None] * 5,
            ],
            inputs=[instruction_input] + image_inputs
        )
    
    return app


if __name__ == "__main__":
    # Run Streamlit app
    create_streamlit_app()
