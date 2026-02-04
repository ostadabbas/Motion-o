"""
Full Token Analysis - Optimized for 32GB GPU
Uses more frames and higher resolution for better results.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import json
import gc

from token_extractor import TokenExtractor
from spatial_analysis import SpatialAnalyzer, create_ball_mask
from temporal_analysis import TemporalAnalyzer


def main():
    print("="*80)
    print("FULL TOKEN ANALYSIS (32GB GPU OPTIMIZED)")
    print("="*80)
    
    # Configuration - Take advantage of available memory
    video_path = "test_videos/Ball_Animation_Video_Generation.mp4"
    num_frames = 60  # Process more frames (was 12-30)
    max_pixels = 720 * 1280  # Higher resolution (was 360×420)
    output_dir = Path("results_full")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Frames: {num_frames}")
    print(f"  Resolution: {max_pixels} pixels")
    print(f"  Output: {output_dir}/")
    
    # Initialize
    extractor = TokenExtractor()
    
    # ========================================================================
    # LOAD FRAMES AND EXTRACT TOKENS
    # ========================================================================
    print(f"\n{'='*60}")
    print("STEP 1: LOADING VIDEO FRAMES")
    print(f"{'='*60}")
    
    frames, fps = extractor.load_video_frames(video_path, num_frames=num_frames)
    print(f"Loaded {len(frames)} frames @ {fps} fps")
    print(f"Frame shape: {frames[0].shape}")
    
    # Create motion masks
    print("Creating motion masks...")
    motion_masks = [create_ball_mask(frame, method='color') for frame in frames]
    motion_coverage = [mask.mean() for mask in motion_masks]
    print(f"Average motion coverage: {100*np.mean(motion_coverage):.1f}%")
    
    # Save sample frames
    print("Saving sample frames...")
    for i in [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, -1]:
        frame_path = output_dir / f"frame_{i:03d}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
        
        overlay = frames[i].copy()
        overlay[motion_masks[i] > 0.5] = [255, 0, 0]
        mask_path = output_dir / f"mask_overlay_{i:03d}.png"
        cv2.imwrite(str(mask_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    # ========================================================================
    # EXTRACT TOKENS FROM ALL FRAMES
    # ========================================================================
    print(f"\n{'='*60}")
    print("STEP 2: EXTRACTING TOKENS")
    print(f"{'='*60}")
    
    all_tokens = []
    
    for i, frame in enumerate(frames):
        if (i + 1) % 10 == 0:
            print(f"  Processing frame {i+1}/{len(frames)}...")
        
        tokens_dict = extractor.extract_tokens_from_frames([frame])
        
        if i == 0:
            for layer_name in tokens_dict.keys():
                all_tokens.append({'layer': layer_name, 'tokens': []})
                print(f"  Layer: {layer_name}, Shape: {tokens_dict[layer_name].shape}")
        
        for j, layer_name in enumerate(tokens_dict.keys()):
            all_tokens[j]['tokens'].append(tokens_dict[layer_name].clone())
        
        torch.cuda.empty_cache()
    
    print(f"Token extraction complete!")
    
    # ========================================================================
    # SPATIAL ANALYSIS
    # ========================================================================
    print(f"\n{'='*60}")
    print("STEP 3: SPATIAL ANALYSIS")
    print(f"{'='*60}")
    
    spatial_analyzer = SpatialAnalyzer(output_dir=str(output_dir / "spatial"))
    spatial_results = {}
    
    # Analyze middle frame
    mid_idx = len(frames) // 2
    mid_frame = frames[mid_idx]
    mid_mask = motion_masks[mid_idx]
    
    for layer_data in all_tokens:
        layer_name = layer_data['layer']
        frame_tokens = layer_data['tokens'][mid_idx]
        
        if frame_tokens.dim() == 3:
            frame_tokens = frame_tokens[0]
        
        print(f"\nAnalyzing layer: {layer_name}")
        
        stats = spatial_analyzer.analyze_token_spatial_structure(
            frame_tokens, mid_frame, layer_name
        )
        
        if 'grid_shape' in stats:
            h, w = stats['grid_shape']
            semantic_stats = spatial_analyzer.compare_semantic_regions(
                frame_tokens, mid_mask, h, w, layer_name
            )
            stats['semantic_comparison'] = semantic_stats
        
        spatial_results[layer_name] = stats
    
    with open(output_dir / "spatial_results.json", 'w') as f:
        json.dump(spatial_results, f, indent=2)
    
    # ========================================================================
    # TEMPORAL ANALYSIS (HIGH-RESOLUTION)
    # ========================================================================
    print(f"\n{'='*60}")
    print("STEP 4: TEMPORAL ANALYSIS (60 FRAMES)")
    print(f"{'='*60}")
    
    temporal_analyzer = TemporalAnalyzer(output_dir=str(output_dir / "temporal"))
    temporal_results = {}
    
    for layer_data in all_tokens:
        layer_name = layer_data['layer']
        token_sequence = torch.stack(layer_data['tokens'])
        
        if token_sequence.dim() == 4:
            token_sequence = token_sequence[:, 0, :, :]
        
        print(f"\nLayer: {layer_name}")
        print(f"Sequence shape: {token_sequence.shape}")
        
        # Compute deltas
        deltas = token_sequence[1:] - token_sequence[:-1]
        
        # Get grid dimensions
        if layer_name in spatial_results and 'grid_shape' in spatial_results[layer_name]:
            h, w = spatial_results[layer_name]['grid_shape']
            
            # Delta magnitude analysis
            delta_results = temporal_analyzer.analyze_delta_magnitude(
                deltas, frames, motion_masks, h, w, layer_name
            )
            
            # Cosine similarity analysis
            sim_results = temporal_analyzer.analyze_cosine_similarity_across_time(
                token_sequence, motion_masks, h, w, layer_name
            )
            
            # Track specific token trajectories
            # Sample tokens from ball path
            ball_token_idx = h * w // 2  # Center token
            traj_results = temporal_analyzer.track_token_trajectory(
                token_sequence, ball_token_idx, layer_name
            )
            
            temporal_results[layer_name] = {
                'delta_analysis': delta_results,
                'similarity_analysis': sim_results,
                'trajectory': traj_results,
            }
    
    with open(output_dir / "temporal_results.json", 'w') as f:
        json.dump(temporal_results, f, indent=2)
    
    # ========================================================================
    # HIGH-RESOLUTION MOTION PROFILE
    # ========================================================================
    print(f"\n{'='*60}")
    print("STEP 5: HIGH-RESOLUTION MOTION PROFILE")
    print(f"{'='*60}")
    
    # With 60 frames, we can create detailed motion profiles
    for layer_data in all_tokens:
        layer_name = layer_data['layer']
        token_sequence = torch.stack(layer_data['tokens'])
        
        if token_sequence.dim() == 4:
            token_sequence = token_sequence[:, 0, :, :]
        
        deltas = token_sequence[1:] - token_sequence[:-1]
        delta_norms = torch.norm(deltas, dim=2).numpy()
        
        # Average delta over space for each time step
        temporal_profile = delta_norms.mean(axis=1)
        
        print(f"\n{layer_name}:")
        print(f"  Mean delta: {temporal_profile.mean():.4f}")
        print(f"  Max delta: {temporal_profile.max():.4f}")
        print(f"  Min delta: {temporal_profile.min():.4f}")
        
        # Plot temporal profile
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14, 4))
        plt.plot(temporal_profile, linewidth=2)
        plt.xlabel('Frame Transition')
        plt.ylabel('Average Delta Magnitude')
        plt.title(f'Temporal Motion Profile - {layer_name} (60 frames)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'motion_profile_{layer_name.replace(".", "_")}.png', dpi=150)
        plt.close()
        print(f"  Saved: motion_profile_{layer_name.replace('.', '_')}.png")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'='*80}")
    print("FULL ANALYSIS SUMMARY (60 FRAMES)")
    print(f"{'='*80}")
    
    print(f"\n1. FRAMES ANALYZED: {len(frames)}")
    print(f"   - Temporal resolution: {fps/len(frames):.2f} fps sampling")
    print(f"   - Motion coverage: {100*np.mean(motion_coverage):.1f}%")
    
    print(f"\n2. SPATIAL STRUCTURE:")
    for layer_name, stats in spatial_results.items():
        if 'grid_shape' in stats:
            h, w = stats['grid_shape']
            print(f"   {layer_name}: {h}×{w} = {h*w} tokens")
    
    print(f"\n3. TEMPORAL DYNAMICS:")
    for layer_name, results in temporal_results.items():
        if 'delta_analysis' in results and results['delta_analysis']:
            ratio = results['delta_analysis'].get('ratio', 0)
            p_val = results['delta_analysis'].get('p_value', 1)
            print(f"   {layer_name}:")
            print(f"     Motion/Static ratio: {ratio:.2f}x")
            print(f"     Significance: p = {p_val:.2e}")
    
    print(f"\nAll results saved to: {output_dir}/")
    
    # Cleanup
    extractor.cleanup()
    del extractor
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n✅ Full analysis complete!")


if __name__ == "__main__":
    main()
