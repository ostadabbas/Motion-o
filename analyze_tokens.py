"""
Simplified Token Analysis
Analyzes visual tokens within memory constraints of small GPU.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import json
import gc
import matplotlib.pyplot as plt

from token_extractor import TokenExtractor
from spatial_analysis import SpatialAnalyzer, create_ball_mask


def main():
    print("="*80)
    print("VISUAL TOKEN ANALYSIS FOR MOTION AUGMENTATION")
    print("="*80)
    
    # Configuration
    video_path = "test_videos/Ball_Animation_Video_Generation.mp4"
    num_frames = 12  # Analyze 12 frames
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize extractor
    print("\nInitializing token extractor...")
    extractor = TokenExtractor()
    
    # ========================================================================
    # STEP 1: Load video frames
    # ========================================================================
    print(f"\n{'='*60}")
    print("STEP 1: LOADING VIDEO FRAMES")
    print(f"{'='*60}")
    
    frames, fps = extractor.load_video_frames(video_path, num_frames=num_frames)
    print(f"Loaded {len(frames)} frames from video")
    
    # Create motion masks for each frame
    print("Creating motion masks (ball detection)...")
    motion_masks = []
    for frame in frames:
        mask = create_ball_mask(frame, method='color')
        motion_masks.append(mask)
    
    # Save sample visualizations
    print("Saving sample frames and masks...")
    for i in [0, len(frames)//2, -1]:
        # Save frame
        frame_path = output_dir / f"frame_{i:03d}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
        
        # Save mask overlay
        overlay = frames[i].copy()
        overlay[motion_masks[i] > 0.5] = [255, 0, 0]  # Red overlay
        mask_path = output_dir / f"mask_overlay_{i:03d}.png"
        cv2.imwrite(str(mask_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    # ========================================================================
    # STEP 2: Extract tokens from multiple frames
    # ========================================================================
    print(f"\n{'='*60}")
    print("STEP 2: EXTRACTING TOKENS FROM FRAMES")
    print(f"{'='*60}")
    
    all_tokens = []
    
    # Process frames one at a time to save memory
    for i, frame in enumerate(frames):
        print(f"\nProcessing frame {i+1}/{len(frames)}...")
        
        tokens_dict = extractor.extract_tokens_from_frames([frame])
        
        # Store tokens for each layer
        if i == 0:
            # Initialize storage
            for layer_name in tokens_dict.keys():
                all_tokens.append({
                    'layer': layer_name,
                    'tokens': []
                })
        
        # Append tokens
        for j, layer_name in enumerate(tokens_dict.keys()):
            all_tokens[j]['tokens'].append(tokens_dict[layer_name].clone())
        
        # Clear memory
        torch.cuda.empty_cache()
    
    print(f"\nCollected tokens from {len(frames)} frames")
    
    # ========================================================================
    # STEP 3: Spatial Analysis (Single Frame)
    # ========================================================================
    print(f"\n{'='*60}")
    print("STEP 3: SPATIAL ANALYSIS")
    print(f"{'='*60}")
    
    spatial_analyzer = SpatialAnalyzer()
    spatial_results = {}
    
    # Analyze middle frame
    mid_idx = len(frames) // 2
    mid_frame = frames[mid_idx]
    mid_mask = motion_masks[mid_idx]
    
    for layer_data in all_tokens:
        layer_name = layer_data['layer']
        frame_tokens = layer_data['tokens'][mid_idx]
        
        # Handle shape
        if frame_tokens.dim() == 3:
            frame_tokens = frame_tokens[0]
        
        print(f"\nAnalyzing layer: {layer_name}")
        print(f"Token shape: {frame_tokens.shape}")
        
        # Spatial structure analysis
        stats = spatial_analyzer.analyze_token_spatial_structure(
            frame_tokens, mid_frame, layer_name
        )
        
        # Semantic region comparison
        if 'grid_shape' in stats:
            h, w = stats['grid_shape']
            semantic_stats = spatial_analyzer.compare_semantic_regions(
                frame_tokens, mid_mask, h, w, layer_name
            )
            stats['semantic_comparison'] = semantic_stats
        
        spatial_results[layer_name] = stats
    
    # Save spatial results
    with open(output_dir / "spatial_results.json", 'w') as f:
        json.dump(spatial_results, f, indent=2)
    
    print(f"\nSpatial analysis complete. Results saved to {output_dir}/spatial_results.json")
    
    # ========================================================================
    # STEP 4: Temporal Analysis (Token Deltas)
    # ========================================================================
    print(f"\n{'='*60}")
    print("STEP 4: TEMPORAL ANALYSIS (DELTAS)")
    print(f"{'='*60}")
    
    temporal_results = {}
    
    for layer_data in all_tokens:
        layer_name = layer_data['layer']
        token_sequence = torch.stack(layer_data['tokens'])  # [num_frames, ...]
        
        # Handle batch dimension
        if token_sequence.dim() == 4:
            token_sequence = token_sequence[:, 0, :, :]  # [num_frames, num_tokens, hidden_dim]
        
        print(f"\nLayer: {layer_name}")
        print(f"Token sequence shape: {token_sequence.shape}")
        
        # Compute deltas
        deltas = token_sequence[1:] - token_sequence[:-1]
        delta_norms = torch.norm(deltas, dim=2).numpy()  # [num_frames-1, num_tokens]
        
        print(f"Delta shape: {deltas.shape}")
        print(f"Delta stats: mean={delta_norms.mean():.6f}, std={delta_norms.std():.6f}")
        
        # Separate by motion/static regions
        motion_deltas = []
        static_deltas = []
        
        # Get grid dimensions
        if layer_name in spatial_results and 'grid_shape' in spatial_results[layer_name]:
            h, w = spatial_results[layer_name]['grid_shape']
            
            for t in range(len(delta_norms)):
                if t >= len(motion_masks):
                    break
                
                # Resize mask
                mask_resized = cv2.resize(
                    motion_masks[t].astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                ).flatten()
                
                # Get delta norms
                frame_deltas = delta_norms[t]
                actual_tokens = min(len(frame_deltas), len(mask_resized))
                
                # Separate
                motion_deltas.extend(frame_deltas[:actual_tokens][mask_resized[:actual_tokens] > 0.5].tolist())
                static_deltas.extend(frame_deltas[:actual_tokens][mask_resized[:actual_tokens] <= 0.5].tolist())
            
            motion_deltas = np.array(motion_deltas)
            static_deltas = np.array(static_deltas)
            
            if len(motion_deltas) > 0 and len(static_deltas) > 0:
                print(f"\nMotion regions: {len(motion_deltas)} samples")
                print(f"  Mean delta: {motion_deltas.mean():.6f} ± {motion_deltas.std():.6f}")
                print(f"Static regions: {len(static_deltas)} samples")
                print(f"  Mean delta: {static_deltas.mean():.6f} ± {static_deltas.std():.6f}")
                print(f"Ratio (motion/static): {motion_deltas.mean()/static_deltas.mean():.2f}x")
                
                # Statistical test
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(motion_deltas, static_deltas)
                print(f"t-test: t={t_stat:.4f}, p={p_value:.4e}")
                
                if p_value < 0.001:
                    print("  *** DELTAS ARE MEANINGFUL! Highly significant correlation with motion ***")
                elif p_value < 0.05:
                    print("  ** Deltas show significant correlation with motion **")
                else:
                    print("  WARNING: Deltas do not correlate strongly with motion")
                
                temporal_results[layer_name] = {
                    'motion_mean': float(motion_deltas.mean()),
                    'motion_std': float(motion_deltas.std()),
                    'static_mean': float(static_deltas.mean()),
                    'static_std': float(static_deltas.std()),
                    'ratio': float(motion_deltas.mean() / static_deltas.mean()),
                    't_stat': float(t_stat),
                    'p_value': float(p_value),
                }
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(static_deltas, bins=50, alpha=0.5, label='Static', color='blue', density=True)
                ax.hist(motion_deltas, bins=50, alpha=0.5, label='Motion', color='red', density=True)
                ax.set_xlabel('Delta L2 Norm')
                ax.set_ylabel('Density')
                ax.set_title(f'Token Delta Distribution: Motion vs Static - {layer_name}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / f'delta_distribution_{layer_name.replace(".", "_")}.png', dpi=150)
                plt.close()
                print(f"  Saved: delta_distribution_{layer_name.replace('.', '_')}.png")
    
    # Save temporal results
    with open(output_dir / "temporal_results.json", 'w') as f:
        json.dump(temporal_results, f, indent=2)
    
    print(f"\nTemporal analysis complete. Results saved to {output_dir}/temporal_results.json")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    print("\nKey Findings:")
    print("\n1. SPATIAL CORRESPONDENCE:")
    for layer_name, stats in spatial_results.items():
        if 'grid_shape' in stats:
            h, w = stats['grid_shape']
            print(f"   {layer_name}: {h}x{w} token grid")
    
    print("\n2. SEMANTIC DIFFERENTIATION:")
    for layer_name, stats in spatial_results.items():
        if 'semantic_comparison' in stats and stats['semantic_comparison']:
            ratio = stats['semantic_comparison'].get('ratio', 0)
            p_val = stats['semantic_comparison'].get('p_value', 1)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.05 else "ns"
            print(f"   {layer_name}: Object/Background ratio = {ratio:.2f}x {sig}")
    
    print("\n3. MOTION ENCODING:")
    for layer_name, stats in temporal_results.items():
        ratio = stats.get('ratio', 0)
        p_val = stats.get('p_value', 1)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.05 else "ns"
        print(f"   {layer_name}: Motion/Static delta ratio = {ratio:.2f}x {sig}")
    
    print(f"\nAll results saved to: {output_dir}/")
    
    # Cleanup
    extractor.cleanup()
    del extractor
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
