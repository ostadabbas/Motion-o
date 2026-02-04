"""
Spatial Analysis of Visual Tokens
Analyzes single frame tokens: spatial mapping, statistics, and semantic correlation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple
import cv2
from pathlib import Path


class SpatialAnalyzer:
    """
    Analyzes spatial structure and properties of visual tokens.
    """
    
    def __init__(self, output_dir: str = "results/spatial"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Spatial analysis output directory: {self.output_dir}")
    
    def analyze_token_spatial_structure(
        self, 
        tokens: torch.Tensor,
        frame: np.ndarray = None,
        layer_name: str = "unknown"
    ) -> Dict:
        """
        Analyze spatial structure of tokens from a single frame.
        
        Args:
            tokens: Token tensor [batch, num_tokens, hidden_dim] or [num_tokens, hidden_dim]
            frame: Original frame for visualization (H, W, 3)
            layer_name: Name of the layer
            
        Returns:
            Dictionary with analysis results
        """
        # Handle batch dimension
        if tokens.dim() == 3:
            tokens = tokens[0]  # Take first batch
        
        num_tokens, hidden_dim = tokens.shape
        
        print(f"\n{'='*60}")
        print(f"SPATIAL ANALYSIS: {layer_name}")
        print(f"{'='*60}")
        print(f"Number of tokens: {num_tokens}")
        print(f"Hidden dimension: {hidden_dim}")
        
        # Infer spatial dimensions (assume square-ish grid)
        # Common patch sizes: 14x14, 16x16, 32x32
        h = w = int(np.sqrt(num_tokens))
        if h * w != num_tokens:
            # Try rectangular grids
            for h_candidate in range(int(np.sqrt(num_tokens)), 0, -1):
                if num_tokens % h_candidate == 0:
                    h = h_candidate
                    w = num_tokens // h
                    break
        
        print(f"Inferred spatial grid: {h} x {w} = {h*w} tokens")
        if h * w != num_tokens:
            print(f"  WARNING: Grid doesn't match token count! Remaining: {num_tokens - h*w}")
        
        # Compute token statistics
        # Replace any NaN or inf values
        tokens_clean = tokens.clone()
        tokens_clean[torch.isnan(tokens_clean)] = 0
        tokens_clean[torch.isinf(tokens_clean)] = 0
        
        token_norms = torch.norm(tokens_clean, dim=1).numpy()  # L2 norm per token
        token_means = tokens_clean.mean(dim=1).numpy()
        token_stds = tokens_clean.std(dim=1).numpy()
        
        # Check for inf/nan in results
        if np.any(np.isnan(token_norms)) or np.any(np.isinf(token_norms)):
            print(f"WARNING: Found NaN or inf in token norms, cleaning...")
            token_norms = np.nan_to_num(token_norms, nan=0.0, posinf=0.0, neginf=0.0)
        
        stats_dict = {
            'num_tokens': num_tokens,
            'hidden_dim': hidden_dim,
            'grid_shape': (h, w),
            'norms': {
                'mean': float(np.mean(token_norms)),
                'std': float(np.std(token_norms)),
                'min': float(np.min(token_norms)),
                'max': float(np.max(token_norms)),
            },
            'means': {
                'mean': float(np.mean(token_means)),
                'std': float(np.std(token_means)),
            },
            'stds': {
                'mean': float(np.mean(token_stds)),
                'std': float(np.std(token_stds)),
            }
        }
        
        print(f"\nToken Norm Statistics:")
        print(f"  Mean: {stats_dict['norms']['mean']:.4f}")
        print(f"  Std:  {stats_dict['norms']['std']:.4f}")
        print(f"  Min:  {stats_dict['norms']['min']:.4f}")
        print(f"  Max:  {stats_dict['norms']['max']:.4f}")
        
        # Visualizations
        self._visualize_token_grid(token_norms, h, w, frame, layer_name)
        self._visualize_token_distribution(token_norms, layer_name)
        
        return stats_dict
    
    def _visualize_token_grid(
        self,
        token_values: np.ndarray,
        h: int,
        w: int,
        frame: np.ndarray,
        layer_name: str
    ):
        """Create heatmap of token values overlaid on frame."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Reshape to grid (use what we can)
        actual_tokens = min(len(token_values), h * w)
        grid_values = token_values[:actual_tokens].reshape(h, w)
        
        # Plot 1: Heatmap
        im = axes[0].imshow(grid_values, cmap='viridis', aspect='auto')
        axes[0].set_title(f'Token L2 Norms - {layer_name}')
        axes[0].set_xlabel('Width (tokens)')
        axes[0].set_ylabel('Height (tokens)')
        plt.colorbar(im, ax=axes[0], label='L2 Norm')
        
        # Plot 2: Overlay on frame if available
        if frame is not None:
            # Resize grid to match frame size
            frame_h, frame_w = frame.shape[:2]
            grid_resized = cv2.resize(grid_values, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
            
            # Normalize for overlay
            grid_norm = (grid_resized - grid_resized.min()) / (grid_resized.max() - grid_resized.min())
            
            # Create overlay
            axes[1].imshow(frame)
            axes[1].imshow(grid_norm, cmap='hot', alpha=0.5)
            axes[1].set_title(f'Token Norms Overlaid on Frame')
            axes[1].axis('off')
        else:
            axes[1].text(0.5, 0.5, 'Frame not available', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].axis('off')
        
        plt.tight_layout()
        save_path = self.output_dir / f'token_grid_{layer_name.replace(".", "_")}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()
    
    def _visualize_token_distribution(self, token_norms: np.ndarray, layer_name: str):
        """Plot distribution of token norms."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        axes[0].hist(token_norms, bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('L2 Norm')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'Distribution of Token Norms - {layer_name}')
        axes[0].axvline(token_norms.mean(), color='red', linestyle='--', label=f'Mean: {token_norms.mean():.2f}')
        axes[0].legend()
        
        # Box plot
        axes[1].boxplot(token_norms, vert=True)
        axes[1].set_ylabel('L2 Norm')
        axes[1].set_title(f'Token Norm Statistics - {layer_name}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / f'token_distribution_{layer_name.replace(".", "_")}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()
    
    def compare_semantic_regions(
        self,
        tokens: torch.Tensor,
        object_mask: np.ndarray,
        h: int,
        w: int,
        layer_name: str = "unknown"
    ) -> Dict:
        """
        Compare tokens from object vs background regions.
        
        Args:
            tokens: Token tensor [num_tokens, hidden_dim]
            object_mask: Binary mask (H, W) where 1 = object, 0 = background
            h, w: Token grid dimensions
            layer_name: Layer name for logging
            
        Returns:
            Dictionary with comparison statistics
        """
        print(f"\n{'='*60}")
        print(f"SEMANTIC REGION COMPARISON: {layer_name}")
        print(f"{'='*60}")
        
        # Handle batch dimension
        if tokens.dim() == 3:
            tokens = tokens[0]
        
        # Resize mask to token grid
        mask_resized = cv2.resize(
            object_mask.astype(np.uint8), 
            (w, h), 
            interpolation=cv2.INTER_NEAREST
        ).flatten()
        
        # Get token norms
        token_norms = torch.norm(tokens, dim=1).numpy()
        
        # Only use tokens that fit the grid
        actual_tokens = min(len(token_norms), h * w, len(mask_resized))
        token_norms = token_norms[:actual_tokens]
        mask_resized = mask_resized[:actual_tokens]
        
        # Split into object and background
        object_norms = token_norms[mask_resized > 0.5]
        background_norms = token_norms[mask_resized <= 0.5]
        
        print(f"Object tokens: {len(object_norms)}")
        print(f"Background tokens: {len(background_norms)}")
        
        if len(object_norms) == 0 or len(background_norms) == 0:
            print("WARNING: One region has no tokens!")
            return {}
        
        # Compute statistics
        object_mean = np.mean(object_norms)
        object_std = np.std(object_norms)
        background_mean = np.mean(background_norms)
        background_std = np.std(background_norms)
        
        print(f"\nObject region:")
        print(f"  Mean norm: {object_mean:.4f} ± {object_std:.4f}")
        print(f"Background region:")
        print(f"  Mean norm: {background_mean:.4f} ± {background_std:.4f}")
        print(f"Ratio (object/background): {object_mean/background_mean:.2f}x")
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(object_norms, background_norms)
        print(f"\nt-test: t={t_stat:.4f}, p={p_value:.4e}")
        
        if p_value < 0.001:
            print("  *** Highly significant difference (p < 0.001)")
        elif p_value < 0.05:
            print("  ** Significant difference (p < 0.05)")
        else:
            print("  No significant difference (p >= 0.05)")
        
        # Cosine similarity within/between regions
        object_tokens = tokens[:actual_tokens][mask_resized > 0.5]
        background_tokens = tokens[:actual_tokens][mask_resized <= 0.5]
        
        # Sample for efficiency
        max_samples = 100
        if len(object_tokens) > max_samples:
            indices = np.random.choice(len(object_tokens), max_samples, replace=False)
            object_tokens = object_tokens[indices]
        if len(background_tokens) > max_samples:
            indices = np.random.choice(len(background_tokens), max_samples, replace=False)
            background_tokens = background_tokens[indices]
        
        # Within-region similarity
        object_sim = self._compute_pairwise_cosine_similarity(object_tokens)
        background_sim = self._compute_pairwise_cosine_similarity(background_tokens)
        
        # Between-region similarity
        between_sim = torch.cosine_similarity(
            object_tokens.unsqueeze(1),
            background_tokens.unsqueeze(0),
            dim=2
        ).numpy().flatten()
        
        print(f"\nCosine Similarity:")
        print(f"  Within object: {object_sim.mean():.4f} ± {object_sim.std():.4f}")
        print(f"  Within background: {background_sim.mean():.4f} ± {background_sim.std():.4f}")
        print(f"  Between regions: {between_sim.mean():.4f} ± {between_sim.std():.4f}")
        
        # Visualization
        self._visualize_semantic_comparison(
            object_norms, background_norms, 
            object_sim, background_sim, between_sim,
            layer_name
        )
        
        return {
            'object_mean': float(object_mean),
            'object_std': float(object_std),
            'background_mean': float(background_mean),
            'background_std': float(background_std),
            'ratio': float(object_mean / background_mean),
            't_stat': float(t_stat),
            'p_value': float(p_value),
            'within_object_sim': float(object_sim.mean()),
            'within_background_sim': float(background_sim.mean()),
            'between_sim': float(between_sim.mean()),
        }
    
    def _compute_pairwise_cosine_similarity(self, tokens: torch.Tensor) -> np.ndarray:
        """Compute pairwise cosine similarity for tokens."""
        # Normalize tokens
        tokens_norm = tokens / tokens.norm(dim=1, keepdim=True)
        # Compute similarity matrix
        sim_matrix = torch.mm(tokens_norm, tokens_norm.t())
        # Get upper triangle (exclude diagonal)
        mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
        similarities = sim_matrix[mask].numpy()
        return similarities
    
    def _visualize_semantic_comparison(
        self,
        object_norms: np.ndarray,
        background_norms: np.ndarray,
        object_sim: np.ndarray,
        background_sim: np.ndarray,
        between_sim: np.ndarray,
        layer_name: str
    ):
        """Visualize comparison between semantic regions."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Norm distributions
        axes[0].hist(background_norms, bins=30, alpha=0.5, label='Background', color='blue')
        axes[0].hist(object_norms, bins=30, alpha=0.5, label='Object', color='red')
        axes[0].axvline(background_norms.mean(), color='blue', linestyle='--', linewidth=2)
        axes[0].axvline(object_norms.mean(), color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Token L2 Norm')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'Token Norms: Object vs Background - {layer_name}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Similarity distributions
        axes[1].hist(object_sim, bins=30, alpha=0.5, label='Within Object', color='red')
        axes[1].hist(background_sim, bins=30, alpha=0.5, label='Within Background', color='blue')
        axes[1].hist(between_sim, bins=30, alpha=0.5, label='Between Regions', color='green')
        axes[1].set_xlabel('Cosine Similarity')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'Token Similarities - {layer_name}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / f'semantic_comparison_{layer_name.replace(".", "_")}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()


def create_ball_mask(frame: np.ndarray, method: str = 'color') -> np.ndarray:
    """
    Create a binary mask for the red ball.
    
    Args:
        frame: RGB frame (H, W, 3)
        method: 'color' for color-based thresholding
        
    Returns:
        Binary mask (H, W) where 1 = ball, 0 = background
    """
    if method == 'color':
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        
        # Red color range (red wraps around in HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return (mask > 0).astype(np.float32)
    
    return np.zeros(frame.shape[:2], dtype=np.float32)


if __name__ == "__main__":
    print("Spatial Analysis Module")
    print("Run from main analysis script")
