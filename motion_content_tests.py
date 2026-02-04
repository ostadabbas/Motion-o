"""
Motion Information Content Tests
Tests what motion information is encoded in token deltas: magnitude, direction, predictive power.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy import stats
from typing import Dict, List, Tuple
import cv2
from pathlib import Path


class MotionContentAnalyzer:
    """
    Analyzes what motion information is encoded in token deltas.
    """
    
    def __init__(self, output_dir: str = "results/motion_content"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Motion content analysis output directory: {self.output_dir}")
    
    def test_predictive_power(
        self,
        deltas: torch.Tensor,
        motion_masks: List[np.ndarray],
        h: int,
        w: int,
        layer_name: str = "unknown"
    ) -> Dict:
        """
        Test if delta magnitude can predict moving objects.
        
        Args:
            deltas: Token deltas [num_frames-1, num_tokens, hidden_dim]
            motion_masks: Ground truth motion masks
            h, w: Grid dimensions
            layer_name: Layer name
            
        Returns:
            Prediction metrics
        """
        print(f"\n{'='*60}")
        print(f"PREDICTIVE POWER TEST: {layer_name}")
        print(f"{'='*60}")
        
        # Compute delta norms
        delta_norms = torch.norm(deltas, dim=2).numpy()  # [num_frames-1, num_tokens]
        
        # Collect all deltas and labels
        all_deltas = []
        all_labels = []
        
        for t in range(min(delta_norms.shape[0], len(motion_masks))):
            # Resize mask
            mask_resized = cv2.resize(
                motion_masks[t].astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST
            ).flatten()
            
            # Get deltas
            frame_deltas = delta_norms[t]
            
            # Only use valid tokens
            actual_tokens = min(len(frame_deltas), len(mask_resized), h * w)
            all_deltas.extend(frame_deltas[:actual_tokens].tolist())
            all_labels.extend(mask_resized[:actual_tokens].tolist())
        
        all_deltas = np.array(all_deltas)
        all_labels = (np.array(all_labels) > 0.5).astype(int)
        
        print(f"Total samples: {len(all_deltas)}")
        print(f"Motion samples: {all_labels.sum()} ({100*all_labels.mean():.1f}%)")
        
        # Test different thresholds
        thresholds = np.percentile(all_deltas, [50, 60, 70, 75, 80, 85, 90, 95])
        
        best_f1 = 0
        best_threshold = 0
        results = []
        
        print(f"\nTesting thresholds:")
        for threshold in thresholds:
            predictions = (all_deltas > threshold).astype(int)
            
            precision = precision_score(all_labels, predictions, zero_division=0)
            recall = recall_score(all_labels, predictions, zero_division=0)
            f1 = f1_score(all_labels, predictions, zero_division=0)
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            })
            
            print(f"  Threshold {threshold:.6f}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        print(f"\nBest threshold: {best_threshold:.6f} (F1={best_f1:.3f})")
        
        # Visualize
        self._visualize_prediction_performance(results, layer_name)
        
        # ROC-style curve
        self._visualize_precision_recall(all_deltas, all_labels, layer_name)
        
        return {
            'best_threshold': float(best_threshold),
            'best_f1': float(best_f1),
            'results': results,
        }
    
    def test_temporal_scaling(
        self,
        token_sequence: torch.Tensor,
        k_values: List[int] = [1, 2, 3, 5, 10],
        layer_name: str = "unknown"
    ) -> Dict:
        """
        Test how deltas scale with temporal distance.
        
        Args:
            token_sequence: Tokens [num_frames, num_tokens, hidden_dim]
            k_values: List of temporal offsets to test
            layer_name: Layer name
            
        Returns:
            Scaling analysis
        """
        print(f"\n{'='*60}")
        print(f"TEMPORAL SCALING TEST: {layer_name}")
        print(f"{'='*60}")
        
        num_frames = token_sequence.shape[0]
        
        scaling_results = []
        
        for k in k_values:
            if k >= num_frames:
                continue
            
            # Compute delta_k
            delta_k = token_sequence[k:] - token_sequence[:-k]
            delta_k_norms = torch.norm(delta_k, dim=2).numpy()
            
            mean_delta = delta_k_norms.mean()
            std_delta = delta_k_norms.std()
            
            scaling_results.append({
                'k': k,
                'mean_delta': mean_delta,
                'std_delta': std_delta,
            })
            
            print(f"k={k}: mean delta = {mean_delta:.6f} ± {std_delta:.6f}")
        
        # Check if scaling is linear
        k_array = np.array([r['k'] for r in scaling_results])
        mean_array = np.array([r['mean_delta'] for r in scaling_results])
        
        # Fit linear model
        slope, intercept = np.polyfit(k_array, mean_array, 1)
        
        # Compute R^2
        y_pred = slope * k_array + intercept
        ss_res = np.sum((mean_array - y_pred) ** 2)
        ss_tot = np.sum((mean_array - mean_array.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        print(f"\nLinear fit: delta(k) = {slope:.6f} * k + {intercept:.6f}")
        print(f"R² = {r_squared:.4f}")
        
        if r_squared > 0.9:
            print("  ✓ Strong linear scaling - deltas encode velocity")
        elif r_squared > 0.7:
            print("  ~ Moderate linear scaling")
        else:
            print("  ✗ Non-linear scaling - deltas may saturate or encode displacement")
        
        # Visualize
        self._visualize_temporal_scaling(scaling_results, slope, intercept, layer_name)
        
        return {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_squared),
            'results': scaling_results,
        }
    
    def test_directional_information(
        self,
        deltas: torch.Tensor,
        motion_masks: List[np.ndarray],
        h: int,
        w: int,
        motion_direction: str = "horizontal",
        layer_name: str = "unknown"
    ) -> Dict:
        """
        Test if deltas encode motion direction.
        
        Args:
            deltas: Token deltas [num_frames-1, num_tokens, hidden_dim]
            motion_masks: Motion masks
            h, w: Grid dimensions
            motion_direction: Expected motion direction ("horizontal" or "vertical")
            layer_name: Layer name
            
        Returns:
            Directional analysis
        """
        print(f"\n{'='*60}")
        print(f"DIRECTIONAL INFORMATION TEST: {layer_name}")
        print(f"{'='*60}")
        print(f"Expected motion direction: {motion_direction}")
        
        # Collect delta vectors from motion regions
        motion_deltas = []
        
        for t in range(min(deltas.shape[0], len(motion_masks))):
            # Resize mask
            mask_resized = cv2.resize(
                motion_masks[t].astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST
            ).flatten()
            
            # Get deltas
            frame_deltas = deltas[t].numpy()
            
            # Only use valid tokens
            actual_tokens = min(len(frame_deltas), len(mask_resized), h * w)
            
            # Extract motion region deltas
            motion_deltas.append(frame_deltas[:actual_tokens][mask_resized[:actual_tokens] > 0.5])
        
        # Concatenate
        motion_deltas = np.vstack(motion_deltas) if motion_deltas else np.array([])
        
        print(f"Collected {len(motion_deltas)} motion delta vectors")
        print(f"Delta vector dimension: {motion_deltas.shape[1] if len(motion_deltas) > 0 else 'N/A'}")
        
        if len(motion_deltas) < 10:
            print("WARNING: Not enough motion samples for PCA")
            return {}
        
        # Perform PCA
        pca = PCA(n_components=min(10, motion_deltas.shape[1]))
        motion_deltas_pca = pca.fit_transform(motion_deltas)
        
        print(f"\nPCA Results:")
        print(f"  PC1 explains {100*pca.explained_variance_ratio_[0]:.1f}% variance")
        print(f"  PC2 explains {100*pca.explained_variance_ratio_[1]:.1f}% variance")
        print(f"  Top 3 PCs explain {100*pca.explained_variance_ratio_[:3].sum():.1f}% variance")
        
        # Check if PC1 is dominant (directional motion)
        if pca.explained_variance_ratio_[0] > 0.5:
            print("  ✓ PC1 is dominant - deltas have strong directional component")
        else:
            print("  ~ PC1 not dominant - motion may be complex or multi-directional")
        
        # Visualize
        self._visualize_pca(motion_deltas_pca, pca, layer_name)
        
        return {
            'pc1_variance': float(pca.explained_variance_ratio_[0]),
            'pc2_variance': float(pca.explained_variance_ratio_[1]),
            'top3_variance': float(pca.explained_variance_ratio_[:3].sum()),
        }
    
    def test_semantic_clustering(
        self,
        deltas: torch.Tensor,
        motion_masks: List[np.ndarray],
        h: int,
        w: int,
        n_clusters: int = 3,
        layer_name: str = "unknown"
    ) -> Dict:
        """
        Test if similar motions cluster together in delta space.
        
        Args:
            deltas: Token deltas
            motion_masks: Motion masks
            h, w: Grid dimensions
            n_clusters: Number of clusters
            layer_name: Layer name
            
        Returns:
            Clustering analysis
        """
        print(f"\n{'='*60}")
        print(f"SEMANTIC CLUSTERING TEST: {layer_name}")
        print(f"{'='*60}")
        
        # Collect motion deltas
        motion_deltas = []
        frame_indices = []
        
        for t in range(min(deltas.shape[0], len(motion_masks))):
            mask_resized = cv2.resize(
                motion_masks[t].astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST
            ).flatten()
            
            frame_deltas = deltas[t].numpy()
            actual_tokens = min(len(frame_deltas), len(mask_resized), h * w)
            
            motion_delta_subset = frame_deltas[:actual_tokens][mask_resized[:actual_tokens] > 0.5]
            
            if len(motion_delta_subset) > 0:
                motion_deltas.append(motion_delta_subset)
                frame_indices.extend([t] * len(motion_delta_subset))
        
        motion_deltas = np.vstack(motion_deltas) if motion_deltas else np.array([])
        frame_indices = np.array(frame_indices)
        
        print(f"Collected {len(motion_deltas)} motion delta vectors")
        
        if len(motion_deltas) < n_clusters:
            print("WARNING: Not enough samples for clustering")
            return {}
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(motion_deltas)
        
        print(f"\nClustering with k={n_clusters}:")
        for i in range(n_clusters):
            count = (cluster_labels == i).sum()
            print(f"  Cluster {i}: {count} samples ({100*count/len(cluster_labels):.1f}%)")
        
        # Analyze clusters by time
        print(f"\nCluster temporal distribution:")
        for i in range(n_clusters):
            cluster_frames = frame_indices[cluster_labels == i]
            if len(cluster_frames) > 0:
                print(f"  Cluster {i}: frames {cluster_frames.min()}-{cluster_frames.max()} (mean: {cluster_frames.mean():.1f})")
        
        # Visualize
        self._visualize_clusters(motion_deltas, cluster_labels, frame_indices, layer_name)
        
        # Inertia (within-cluster sum of squares)
        inertia = kmeans.inertia_
        print(f"\nInertia: {inertia:.2f}")
        
        return {
            'n_clusters': n_clusters,
            'inertia': float(inertia),
            'cluster_sizes': [int((cluster_labels == i).sum()) for i in range(n_clusters)],
        }
    
    def _visualize_prediction_performance(self, results: List[Dict], layer_name: str):
        """Visualize precision/recall/F1 vs threshold."""
        thresholds = [r['threshold'] for r in results]
        precisions = [r['precision'] for r in results]
        recalls = [r['recall'] for r in results]
        f1s = [r['f1'] for r in results]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(thresholds, precisions, marker='o', label='Precision', linewidth=2)
        ax.plot(thresholds, recalls, marker='s', label='Recall', linewidth=2)
        ax.plot(thresholds, f1s, marker='^', label='F1 Score', linewidth=2)
        
        ax.set_xlabel('Delta Threshold')
        ax.set_ylabel('Score')
        ax.set_title(f'Motion Detection Performance - {layer_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / f'prediction_performance_{layer_name.replace(".", "_")}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()
    
    def _visualize_precision_recall(self, deltas: np.ndarray, labels: np.ndarray, layer_name: str):
        """Visualize precision-recall curve."""
        # Sort by delta
        sorted_idx = np.argsort(deltas)[::-1]
        deltas_sorted = deltas[sorted_idx]
        labels_sorted = labels[sorted_idx]
        
        # Compute cumulative precision and recall
        precisions = []
        recalls = []
        
        for i in range(1, len(deltas_sorted), max(1, len(deltas_sorted) // 100)):
            predictions = np.zeros(len(labels_sorted))
            predictions[:i] = 1
            
            tp = np.sum((predictions == 1) & (labels_sorted == 1))
            fp = np.sum((predictions == 1) & (labels_sorted == 0))
            fn = np.sum((predictions == 0) & (labels_sorted == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.plot(recalls, precisions, linewidth=2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve - {layer_name}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        save_path = self.output_dir / f'precision_recall_curve_{layer_name.replace(".", "_")}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()
    
    def _visualize_temporal_scaling(self, results: List[Dict], slope: float, intercept: float, layer_name: str):
        """Visualize temporal scaling."""
        k_values = [r['k'] for r in results]
        means = [r['mean_delta'] for r in results]
        stds = [r['std_delta'] for r in results]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.errorbar(k_values, means, yerr=stds, fmt='o', markersize=8, capsize=5, label='Data')
        
        # Plot linear fit
        k_fit = np.array(k_values)
        y_fit = slope * k_fit + intercept
        ax.plot(k_fit, y_fit, '--', linewidth=2, label=f'Linear fit: {slope:.4f}k + {intercept:.4f}')
        
        ax.set_xlabel('Temporal Offset (k frames)')
        ax.set_ylabel('Mean Delta L2 Norm')
        ax.set_title(f'Temporal Scaling of Deltas - {layer_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / f'temporal_scaling_{layer_name.replace(".", "_")}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()
    
    def _visualize_pca(self, deltas_pca: np.ndarray, pca: PCA, layer_name: str):
        """Visualize PCA of motion deltas."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot PC1 vs PC2
        axes[0].scatter(deltas_pca[:, 0], deltas_pca[:, 1], alpha=0.5, s=10)
        axes[0].set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)')
        axes[0].set_ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)')
        axes[0].set_title(f'PCA of Motion Deltas - {layer_name}')
        axes[0].grid(True, alpha=0.3)
        
        # Scree plot
        axes[1].bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
        axes[1].set_xlabel('Principal Component')
        axes[1].set_ylabel('Explained Variance Ratio')
        axes[1].set_title(f'PCA Scree Plot - {layer_name}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / f'pca_analysis_{layer_name.replace(".", "_")}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()
    
    def _visualize_clusters(self, deltas: np.ndarray, labels: np.ndarray, frame_indices: np.ndarray, layer_name: str):
        """Visualize clustering results."""
        # PCA for visualization
        pca = PCA(n_components=2)
        deltas_2d = pca.fit_transform(deltas)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Clusters in PCA space
        scatter = axes[0].scatter(deltas_2d[:, 0], deltas_2d[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
        axes[0].set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)')
        axes[0].set_ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)')
        axes[0].set_title(f'Motion Delta Clusters - {layer_name}')
        plt.colorbar(scatter, ax=axes[0], label='Cluster')
        
        # Plot 2: Clusters over time
        axes[1].scatter(frame_indices, labels, c=labels, cmap='tab10', alpha=0.6, s=20)
        axes[1].set_xlabel('Frame Index')
        axes[1].set_ylabel('Cluster')
        axes[1].set_title(f'Cluster Temporal Distribution - {layer_name}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / f'clustering_{layer_name.replace(".", "_")}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()


if __name__ == "__main__":
    print("Motion Content Tests Module")
    print("Run from main analysis script")
