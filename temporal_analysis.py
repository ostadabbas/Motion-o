"""
Temporal Analysis of Visual Tokens
Analyzes token dynamics across frames to understand motion encoding.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
from typing import Dict, List, Tuple
import cv2
from pathlib import Path


class TemporalAnalyzer:
    """
    Analyzes how visual tokens change across video frames.
    """
    
    def __init__(self, output_dir: str = "results/temporal"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Temporal analysis output directory: {self.output_dir}")
    
    def compute_token_deltas(
        self,
        token_sequence: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute frame-to-frame token deltas.
        
        Args:
            token_sequence: Tokens from all frames [num_frames, num_tokens, hidden_dim]
            
        Returns:
            Deltas [num_frames-1, num_tokens, hidden_dim]
        """
        print(f"\nComputing token deltas...")
        print(f"Input shape: {token_sequence.shape}")
        
        # Compute differences
        deltas = token_sequence[1:] - token_sequence[:-1]
        
        print(f"Delta shape: {deltas.shape}")
        print(f"Delta stats: mean={deltas.mean():.6f}, std={deltas.std():.6f}")
        
        return deltas
    
    def analyze_delta_magnitude(
        self,
        deltas: torch.Tensor,
        frames: List[np.ndarray],
        motion_masks: List[np.ndarray],
        h: int,
        w: int,
        layer_name: str = "unknown"
    ) -> Dict:
        """
        Analyze delta magnitude and correlation with motion.
        
        Args:
            deltas: Token deltas [num_frames-1, num_tokens, hidden_dim]
            frames: List of video frames
            motion_masks: List of binary masks indicating motion regions
            h, w: Token grid dimensions
            layer_name: Layer name
            
        Returns:
            Analysis results dictionary
        """
        print(f"\n{'='*60}")
        print(f"DELTA MAGNITUDE ANALYSIS: {layer_name}")
        print(f"{'='*60}")
        
        num_deltas = deltas.shape[0]
        
        # Compute L2 norm of each delta
        delta_norms = torch.norm(deltas, dim=2)  # [num_frames-1, num_tokens]
        print(f"Delta norms shape: {delta_norms.shape}")
        
        # Create spatiotemporal heatmap
        self._visualize_spatiotemporal_heatmap(delta_norms, h, w, layer_name)
        
        # Correlate with motion masks
        motion_delta_norms = []
        static_delta_norms = []
        
        for t in range(num_deltas):
            if t >= len(motion_masks):
                break
            
            # Resize motion mask to token grid
            mask_resized = cv2.resize(
                motion_masks[t].astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST
            ).flatten()
            
            # Get delta norms for this frame
            frame_deltas = delta_norms[t].numpy()
            
            # Only use tokens that fit
            actual_tokens = min(len(frame_deltas), len(mask_resized), h * w)
            frame_deltas = frame_deltas[:actual_tokens]
            mask_resized = mask_resized[:actual_tokens]
            
            # Separate by motion/static
            motion_delta_norms.extend(frame_deltas[mask_resized > 0.5].tolist())
            static_delta_norms.extend(frame_deltas[mask_resized <= 0.5].tolist())
        
        motion_delta_norms = np.array(motion_delta_norms)
        static_delta_norms = np.array(static_delta_norms)
        
        print(f"\nMotion region deltas: {len(motion_delta_norms)} samples")
        print(f"Static region deltas: {len(static_delta_norms)} samples")
        
        if len(motion_delta_norms) > 0 and len(static_delta_norms) > 0:
            motion_mean = motion_delta_norms.mean()
            motion_std = motion_delta_norms.std()
            static_mean = static_delta_norms.mean()
            static_std = static_delta_norms.std()
            
            print(f"\nMotion regions:")
            print(f"  Delta magnitude: {motion_mean:.6f} ± {motion_std:.6f}")
            print(f"Static regions:")
            print(f"  Delta magnitude: {static_mean:.6f} ± {static_std:.6f}")
            print(f"Ratio (motion/static): {motion_mean/static_mean:.2f}x")
            
            # Statistical test
            t_stat, p_value = stats.ttest_ind(motion_delta_norms, static_delta_norms)
            print(f"\nt-test: t={t_stat:.4f}, p={p_value:.4e}")
            
            if p_value < 0.001:
                print("  *** Highly significant difference (p < 0.001)")
                print("  CONCLUSION: Deltas are significantly larger in motion regions!")
            elif p_value < 0.05:
                print("  ** Significant difference (p < 0.05)")
            else:
                print("  No significant difference (p >= 0.05)")
                print("  WARNING: Deltas may not correlate with motion!")
            
            # Signal-to-noise ratio
            snr = motion_mean / static_mean if static_mean > 0 else 0
            print(f"\nSignal-to-noise ratio: {snr:.2f}")
            
            # Visualize distributions
            self._visualize_motion_static_comparison(
                motion_delta_norms, static_delta_norms, layer_name
            )
            
            results = {
                'motion_mean': float(motion_mean),
                'motion_std': float(motion_std),
                'static_mean': float(static_mean),
                'static_std': float(static_std),
                'ratio': float(motion_mean / static_mean),
                'snr': float(snr),
                't_stat': float(t_stat),
                'p_value': float(p_value),
            }
        else:
            print("WARNING: Not enough samples for comparison")
            results = {}
        
        return results
    
    def analyze_cosine_similarity_across_time(
        self,
        token_sequence: torch.Tensor,
        motion_masks: List[np.ndarray],
        h: int,
        w: int,
        layer_name: str = "unknown"
    ) -> Dict:
        """
        Analyze cosine similarity between consecutive frames.
        
        Args:
            token_sequence: Tokens [num_frames, num_tokens, hidden_dim]
            motion_masks: Motion masks for each frame
            h, w: Grid dimensions
            layer_name: Layer name
            
        Returns:
            Analysis results
        """
        print(f"\n{'='*60}")
        print(f"COSINE SIMILARITY ANALYSIS: {layer_name}")
        print(f"{'='*60}")
        
        num_frames = token_sequence.shape[0]
        
        # Compute cosine similarity between consecutive frames
        motion_similarities = []
        static_similarities = []
        
        for t in range(num_frames - 1):
            tokens_t = token_sequence[t]
            tokens_t1 = token_sequence[t + 1]
            
            # Compute cosine similarity for each token
            cos_sim = torch.cosine_similarity(tokens_t, tokens_t1, dim=1).numpy()
            
            if t < len(motion_masks):
                # Resize mask
                mask_resized = cv2.resize(
                    motion_masks[t].astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                ).flatten()
                
                # Only use valid tokens
                actual_tokens = min(len(cos_sim), len(mask_resized), h * w)
                cos_sim = cos_sim[:actual_tokens]
                mask_resized = mask_resized[:actual_tokens]
                
                # Separate by motion/static
                motion_similarities.extend(cos_sim[mask_resized > 0.5].tolist())
                static_similarities.extend(cos_sim[mask_resized <= 0.5].tolist())
        
        motion_similarities = np.array(motion_similarities)
        static_similarities = np.array(static_similarities)
        
        print(f"\nMotion region similarities: {len(motion_similarities)} samples")
        print(f"Static region similarities: {len(static_similarities)} samples")
        
        if len(motion_similarities) > 0 and len(static_similarities) > 0:
            motion_mean = motion_similarities.mean()
            static_mean = static_similarities.mean()
            
            print(f"\nMotion regions:")
            print(f"  Cosine similarity: {motion_mean:.6f} ± {motion_similarities.std():.6f}")
            print(f"Static regions:")
            print(f"  Cosine similarity: {static_mean:.6f} ± {static_similarities.std():.6f}")
            
            if static_mean > motion_mean:
                print(f"  Static regions are {static_mean/motion_mean:.2f}x more similar across time")
                print("  ✓ This is expected: static regions should be more consistent")
            
            # Visualization
            self._visualize_similarity_comparison(
                motion_similarities, static_similarities, layer_name
            )
            
            return {
                'motion_similarity_mean': float(motion_mean),
                'static_similarity_mean': float(static_mean),
            }
        
        return {}
    
    def track_token_trajectory(
        self,
        token_sequence: torch.Tensor,
        token_idx: int,
        layer_name: str = "unknown"
    ) -> Dict:
        """
        Track a specific token across time.
        
        Args:
            token_sequence: Tokens [num_frames, num_tokens, hidden_dim]
            token_idx: Index of token to track
            layer_name: Layer name
            
        Returns:
            Trajectory analysis
        """
        print(f"\nTracking token {token_idx} across time...")
        
        num_frames = token_sequence.shape[0]
        token_trajectory = token_sequence[:, token_idx, :]  # [num_frames, hidden_dim]
        
        # Compute norms over time
        norms = torch.norm(token_trajectory, dim=1).numpy()
        
        # Compute deltas
        deltas = token_trajectory[1:] - token_trajectory[:-1]
        delta_norms = torch.norm(deltas, dim=1).numpy()
        
        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        axes[0].plot(norms, marker='o', linewidth=2)
        axes[0].set_xlabel('Frame')
        axes[0].set_ylabel('Token L2 Norm')
        axes[0].set_title(f'Token {token_idx} Norm Trajectory - {layer_name}')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(delta_norms, marker='o', color='red', linewidth=2)
        axes[1].set_xlabel('Frame Transition')
        axes[1].set_ylabel('Delta L2 Norm')
        axes[1].set_title(f'Token {token_idx} Delta Magnitude - {layer_name}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / f'token_trajectory_{token_idx}_{layer_name.replace(".", "_")}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()
        
        return {
            'mean_norm': float(norms.mean()),
            'mean_delta': float(delta_norms.mean()),
            'max_delta': float(delta_norms.max()),
        }
    
    def _visualize_spatiotemporal_heatmap(
        self,
        delta_norms: torch.Tensor,
        h: int,
        w: int,
        layer_name: str
    ):
        """Create spatiotemporal heatmap of delta magnitudes."""
        # delta_norms: [num_frames-1, num_tokens]
        num_frames = delta_norms.shape[0]
        
        # Reshape to spatial grid
        actual_tokens = min(delta_norms.shape[1], h * w)
        delta_grid = delta_norms[:, :actual_tokens].numpy()
        delta_grid = delta_grid.reshape(num_frames, h, w)
        
        # Flatten spatial dimensions for heatmap
        delta_flat = delta_grid.reshape(num_frames, h * w)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Time x Space heatmap
        im1 = axes[0].imshow(delta_flat.T, aspect='auto', cmap='hot', interpolation='nearest')
        axes[0].set_xlabel('Frame Transition')
        axes[0].set_ylabel('Token Index (spatial position)')
        axes[0].set_title(f'Spatiotemporal Delta Heatmap - {layer_name}')
        plt.colorbar(im1, ax=axes[0], label='Delta L2 Norm')
        
        # Plot 2: Average spatial delta across time
        avg_spatial_delta = delta_flat.mean(axis=0).reshape(h, w)
        im2 = axes[1].imshow(avg_spatial_delta, cmap='hot', aspect='auto')
        axes[1].set_xlabel('Width (tokens)')
        axes[1].set_ylabel('Height (tokens)')
        axes[1].set_title(f'Average Delta Magnitude (across time) - {layer_name}')
        plt.colorbar(im2, ax=axes[1], label='Mean Delta L2 Norm')
        
        plt.tight_layout()
        save_path = self.output_dir / f'spatiotemporal_heatmap_{layer_name.replace(".", "_")}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()
    
    def _visualize_motion_static_comparison(
        self,
        motion_deltas: np.ndarray,
        static_deltas: np.ndarray,
        layer_name: str
    ):
        """Visualize comparison of deltas in motion vs static regions."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(static_deltas, bins=50, alpha=0.6, label='Static', color='blue', density=True)
        axes[0].hist(motion_deltas, bins=50, alpha=0.6, label='Motion', color='red', density=True)
        axes[0].axvline(static_deltas.mean(), color='blue', linestyle='--', linewidth=2, label=f'Static mean: {static_deltas.mean():.4f}')
        axes[0].axvline(motion_deltas.mean(), color='red', linestyle='--', linewidth=2, label=f'Motion mean: {motion_deltas.mean():.4f}')
        axes[0].set_xlabel('Delta L2 Norm')
        axes[0].set_ylabel('Density')
        axes[0].set_title(f'Delta Distribution: Motion vs Static - {layer_name}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot([static_deltas, motion_deltas], labels=['Static', 'Motion'])
        axes[1].set_ylabel('Delta L2 Norm')
        axes[1].set_title(f'Delta Statistics - {layer_name}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / f'motion_static_delta_comparison_{layer_name.replace(".", "_")}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()
    
    def _visualize_similarity_comparison(
        self,
        motion_sim: np.ndarray,
        static_sim: np.ndarray,
        layer_name: str
    ):
        """Visualize cosine similarity comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(static_sim, bins=50, alpha=0.6, label='Static', color='blue', density=True)
        ax.hist(motion_sim, bins=50, alpha=0.6, label='Motion', color='red', density=True)
        ax.axvline(static_sim.mean(), color='blue', linestyle='--', linewidth=2)
        ax.axvline(motion_sim.mean(), color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Cosine Similarity (consecutive frames)')
        ax.set_ylabel('Density')
        ax.set_title(f'Token Consistency: Motion vs Static - {layer_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / f'similarity_comparison_{layer_name.replace(".", "_")}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()


if __name__ == "__main__":
    print("Temporal Analysis Module")
    print("Run from main analysis script")
