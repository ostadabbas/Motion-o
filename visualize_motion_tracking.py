"""
Visualize Motion Tracking in Qwen3-VL Visual Tokens

Shows how token deltas and embeddings track the moving ball across frames.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import gc

from token_extractor_qwen3 import TokenExtractorQwen3
from spatial_analysis import create_ball_mask


def find_ball_position(mask):
    """Find the centroid of the ball in the mask."""
    if mask.sum() == 0:
        return None
    y_coords, x_coords = np.where(mask > 0.5)
    return int(y_coords.mean()), int(x_coords.mean())


def map_to_token_grid(pixel_y, pixel_x, img_h, img_w, grid_h, grid_w):
    """Map pixel coordinates to token grid coordinates."""
    token_y = int(pixel_y * grid_h / img_h)
    token_x = int(pixel_x * grid_w / img_w)
    return token_y, token_x


def get_ball_tokens(tokens, mask, grid_h, grid_w):
    """Extract tokens corresponding to the ball region."""
    img_h, img_w = mask.shape
    mask_resized = cv2.resize(
        mask.astype(np.uint8), (grid_w, grid_h),
        interpolation=cv2.INTER_NEAREST
    )
    
    # Get indices of ball tokens
    ball_token_indices = np.where(mask_resized.flatten() > 0.5)[0]
    
    if len(ball_token_indices) == 0:
        return None, None
    
    # Extract ball tokens
    if tokens.dim() == 3:
        tokens = tokens[0]  # Remove batch dim if present
    
    # Safety check: ensure indices are within bounds
    max_idx = tokens.shape[0]
    ball_token_indices = ball_token_indices[ball_token_indices < max_idx]
    
    if len(ball_token_indices) == 0:
        return None, None
    
    ball_tokens = tokens[ball_token_indices]
    return ball_tokens, ball_token_indices


def visualize_delta_heatmap_sequence(frames, deltas, masks, grid_h, grid_w, output_path, layer_name):
    """Create a sequence showing frame + delta heatmap overlaid."""
    n_frames = min(len(frames), len(deltas) + 1)
    
    # Select frames to show (evenly spaced)
    indices = np.linspace(0, len(deltas) - 1, min(8, len(deltas)), dtype=int)
    
    fig, axes = plt.subplots(2, len(indices), figsize=(20, 8))
    
    for idx, i in enumerate(indices):
        # Top row: Original frame with ball highlighted
        ax_frame = axes[0, idx]
        frame = frames[i].copy()
        overlay = frame.copy()
        overlay[masks[i] > 0.5] = [255, 0, 0]
        combined = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        ax_frame.imshow(combined)
        ax_frame.set_title(f'Frame {i}')
        ax_frame.axis('off')
        
        # Bottom row: Delta heatmap
        ax_delta = axes[1, idx]
        delta_norms = torch.norm(deltas[i], dim=1).numpy()
        
        # Verify size before reshape
        if len(delta_norms) != grid_h * grid_w:
            print(f"    Warning: delta size {len(delta_norms)} != grid {grid_h}x{grid_w}, truncating...")
            delta_norms = delta_norms[:grid_h * grid_w]
        
        delta_heatmap = delta_norms.reshape(grid_h, grid_w)
        
        # Resize to match frame
        delta_heatmap_resized = cv2.resize(delta_heatmap, (frame.shape[1], frame.shape[0]))
        
        im = ax_delta.imshow(delta_heatmap_resized, cmap='hot', alpha=0.7)
        ax_delta.imshow(frame, alpha=0.3)
        ax_delta.set_title(f'Delta Magnitude')
        ax_delta.axis('off')
    
    plt.suptitle(f'Motion Tracking via Token Deltas - {layer_name}', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_ball_trajectory(frames, all_tokens, masks, grid_h, grid_w, output_path, layer_name):
    """Visualize the ball's trajectory in token embedding space using PCA."""
    print(f"  Computing PCA trajectory for {layer_name}...")
    
    ball_positions = []
    ball_token_embeddings = []
    valid_frames = []
    
    for i, (tokens, mask) in enumerate(zip(all_tokens, masks)):
        pos = find_ball_position(mask)
        if pos is None:
            continue
        
        ball_tokens, _ = get_ball_tokens(tokens, mask, grid_h, grid_w)
        if ball_tokens is None or len(ball_tokens) == 0:
            continue
        
        # Average the ball tokens to get a single embedding per frame
        avg_ball_embedding = ball_tokens.mean(dim=0).numpy()
        
        ball_positions.append(pos)
        ball_token_embeddings.append(avg_ball_embedding)
        valid_frames.append(i)
    
    if len(ball_token_embeddings) < 3:
        print(f"  Not enough ball tokens found for PCA")
        return
    
    # PCA to 2D
    embeddings_array = np.array(ball_token_embeddings)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_array)
    
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_scaled)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Ball position in pixel space
    ax1 = axes[0]
    pixel_positions = np.array(ball_positions)
    scatter1 = ax1.scatter(pixel_positions[:, 1], pixel_positions[:, 0], 
                          c=valid_frames, cmap='viridis', s=100)
    ax1.plot(pixel_positions[:, 1], pixel_positions[:, 0], 'r--', alpha=0.3, linewidth=2)
    ax1.set_xlabel('X Position (pixels)')
    ax1.set_ylabel('Y Position (pixels)')
    ax1.set_title('Ball Trajectory (Pixel Space)')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Frame')
    
    # 2. Ball embedding trajectory in PCA space
    ax2 = axes[1]
    scatter2 = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                          c=valid_frames, cmap='viridis', s=100)
    ax2.plot(embeddings_2d[:, 0], embeddings_2d[:, 1], 'r--', alpha=0.3, linewidth=2)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax2.set_title('Ball Trajectory (Token Embedding Space)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Frame')
    
    # 3. Token distance from initial position
    ax3 = axes[2]
    initial_embedding = embeddings_array[0]
    distances = np.linalg.norm(embeddings_array - initial_embedding, axis=1)
    ax3.plot(valid_frames, distances, 'b-', linewidth=2, marker='o')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('L2 Distance from Initial Token')
    ax3.set_title('Token Embedding Change Over Time')
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f'Ball Motion Tracking in Token Space - {layer_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_delta_vs_motion(frames, deltas, masks, grid_h, grid_w, output_path, layer_name):
    """Show correlation between delta magnitude and actual motion."""
    print(f"  Computing delta vs motion correlation for {layer_name}...")
    
    frame_motion_magnitudes = []
    frame_delta_magnitudes = []
    
    for i in range(len(deltas)):
        if i + 1 >= len(masks):
            break
        
        # Compute actual motion (pixel displacement)
        pos1 = find_ball_position(masks[i])
        pos2 = find_ball_position(masks[i + 1])
        
        if pos1 is None or pos2 is None:
            continue
        
        pixel_motion = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
        
        # Get ball tokens and their delta magnitude
        ball_tokens_delta, _ = get_ball_tokens(deltas[i], masks[i], grid_h, grid_w)
        
        if ball_tokens_delta is None or len(ball_tokens_delta) == 0:
            continue
        
        avg_delta_magnitude = torch.norm(ball_tokens_delta, dim=1).mean().item()
        
        frame_motion_magnitudes.append(pixel_motion)
        frame_delta_magnitudes.append(avg_delta_magnitude)
    
    if len(frame_motion_magnitudes) < 2:
        print(f"  Not enough data for correlation")
        return
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Scatter plot
    ax1 = axes[0]
    ax1.scatter(frame_motion_magnitudes, frame_delta_magnitudes, s=100, alpha=0.6)
    
    # Add correlation
    correlation = np.corrcoef(frame_motion_magnitudes, frame_delta_magnitudes)[0, 1]
    
    # Fit line
    z = np.polyfit(frame_motion_magnitudes, frame_delta_magnitudes, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(frame_motion_magnitudes), max(frame_motion_magnitudes), 100)
    ax1.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    
    ax1.set_xlabel('Actual Motion (pixels)', fontsize=12)
    ax1.set_ylabel('Token Delta Magnitude', fontsize=12)
    ax1.set_title(f'Motion vs Delta Correlation\nr = {correlation:.3f}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 2. Time series
    ax2 = axes[1]
    frames_axis = range(len(frame_motion_magnitudes))
    
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(frames_axis, frame_motion_magnitudes, 'b-o', 
                     label='Pixel Motion', linewidth=2, markersize=6)
    line2 = ax2_twin.plot(frames_axis, frame_delta_magnitudes, 'r-s', 
                          label='Token Delta', linewidth=2, markersize=6)
    
    ax2.set_xlabel('Frame', fontsize=12)
    ax2.set_ylabel('Pixel Motion (pixels)', fontsize=12, color='b')
    ax2_twin.set_ylabel('Token Delta Magnitude', fontsize=12, color='r')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    ax2.set_title('Motion Tracking Over Time', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    
    plt.suptitle(f'Token Deltas Track Physical Motion - {layer_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    print("="*80)
    print("MOTION TRACKING VISUALIZATION - Qwen3-VL")
    print("="*80)
    
    video_path = "test_videos/Ball_Animation_Video_Generation.mp4"
    num_frames = 30
    output_dir = Path("results_qwen3/motion_tracking")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nConfiguration:")
    print(f"  Frames: {num_frames}")
    print(f"  Output: {output_dir}/")
    
    # Load model and extract tokens
    print(f"\n{'='*60}")
    print("LOADING MODEL AND EXTRACTING TOKENS")
    print(f"{'='*60}")
    
    extractor = TokenExtractorQwen3()
    frames, fps = extractor.load_video_frames(video_path, num_frames=num_frames)
    print(f"Loaded {len(frames)} frames @ {fps} fps")
    
    # Create motion masks
    print("Creating motion masks...")
    motion_masks = [create_ball_mask(frame, method='color') for frame in frames]
    
    # Extract tokens
    print("Extracting tokens from all frames...")
    all_tokens = {}
    
    for i, frame in enumerate(frames):
        if (i + 1) % 5 == 0:
            print(f"  Frame {i+1}/{len(frames)}...")
        
        tokens_dict = extractor.extract_tokens_from_frames([frame])
        
        for layer_name, tokens in tokens_dict.items():
            if layer_name not in all_tokens:
                all_tokens[layer_name] = []
            all_tokens[layer_name].append(tokens.clone())
        
        torch.cuda.empty_cache()
    
    # Process each layer
    print(f"\n{'='*60}")
    print("CREATING MOTION TRACKING VISUALIZATIONS")
    print(f"{'='*60}")
    
    # Grid shapes from spatial analysis
    grid_shapes = {
        'siglip2.blocks.last': (46, 80),
        'visual.merger': (23, 40)
    }
    
    for layer_name, token_list in all_tokens.items():
        print(f"\n{layer_name}:")
        
        # Stack tokens
        tokens_stacked = torch.stack(token_list)
        if tokens_stacked.dim() == 4:
            tokens_stacked = tokens_stacked[:, 0, :, :]  # Remove batch dim
        
        # Get grid shape from known values
        num_tokens = tokens_stacked.shape[1]
        if layer_name in grid_shapes:
            grid_h, grid_w = grid_shapes[layer_name]
        else:
            # Fallback: compute from aspect ratio
            img_h, img_w = frames[0].shape[:2]
            aspect = img_w / img_h
            grid_h = int(np.sqrt(num_tokens / aspect))
            grid_w = num_tokens // grid_h
        
        print(f"  Token grid: {grid_h}x{grid_w} = {num_tokens} (expected: {grid_h*grid_w})")
        
        # Verify the calculation
        if grid_h * grid_w != num_tokens:
            print(f"  ⚠️  Grid mismatch! Adjusting...")
            # Find correct factorization
            for h in range(int(np.sqrt(num_tokens)), 0, -1):
                if num_tokens % h == 0:
                    grid_h = h
                    grid_w = num_tokens // h
                    break
            print(f"  Corrected grid: {grid_h}x{grid_w} = {grid_h*grid_w}")
        
        # Compute deltas
        deltas = tokens_stacked[1:] - tokens_stacked[:-1]
        
        # 1. Delta heatmap sequence
        print(f"  Creating delta heatmap sequence...")
        visualize_delta_heatmap_sequence(
            frames, deltas, motion_masks, grid_h, grid_w,
            output_dir / f"delta_heatmap_sequence_{layer_name.replace('.', '_')}.png",
            layer_name
        )
        
        # 2. Ball trajectory in embedding space
        print(f"  Creating ball trajectory visualization...")
        visualize_ball_trajectory(
            frames, token_list, motion_masks, grid_h, grid_w,
            output_dir / f"ball_trajectory_{layer_name.replace('.', '_')}.png",
            layer_name
        )
        
        # 3. Delta vs actual motion correlation
        print(f"  Creating delta vs motion correlation...")
        visualize_delta_vs_motion(
            frames, deltas, motion_masks, grid_h, grid_w,
            output_dir / f"delta_vs_motion_{layer_name.replace('.', '_')}.png",
            layer_name
        )
    
    # Cleanup
    extractor.cleanup()
    del extractor
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\n✅ All visualizations saved to: {output_dir}/")
    print(f"\nWhat to look for:")
    print(f"  1. Delta heatmaps: Hot spots should follow the ball")
    print(f"  2. Ball trajectory: Smooth path in embedding space = good tracking")
    print(f"  3. Delta vs motion: High correlation = tokens encode motion magnitude")
    print(f"\nThese visualizations show if visual tokens can TRACK motion!")


if __name__ == "__main__":
    main()
