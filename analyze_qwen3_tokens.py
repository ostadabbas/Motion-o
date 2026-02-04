"""
Qwen3-VL Token Analysis with SigLIP2
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import json
import gc

from token_extractor_qwen3 import TokenExtractorQwen3
from spatial_analysis import SpatialAnalyzer, create_ball_mask


def main():
    print("="*80)
    print("QWEN3-VL (SigLIP2) TOKEN ANALYSIS")
    print("="*80)
    
    video_path = "test_videos/Ball_Animation_Video_Generation.mp4"
    num_frames = 30  # Analyze 30 frames
    output_dir = Path("results_qwen3")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Model: Qwen3-VL-8B-Instruct (with SigLIP2)")
    print(f"  Frames: {num_frames}")
    print(f"  Output: {output_dir}/")
    
    # Initialize Qwen3-VL extractor
    extractor = TokenExtractorQwen3()
    
    # Load frames
    print(f"\n{'='*60}")
    print("LOADING VIDEO FRAMES")
    print(f"{'='*60}")
    
    frames, fps = extractor.load_video_frames(video_path, num_frames=num_frames)
    print(f"Loaded {len(frames)} frames @ {fps} fps")
    
    # Create motion masks
    print("Creating motion masks...")
    motion_masks = [create_ball_mask(frame, method='color') for frame in frames]
    
    # Save sample frames
    for i in [0, len(frames)//2, -1]:
        cv2.imwrite(str(output_dir / f"frame_{i:03d}.png"), 
                    cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
        overlay = frames[i].copy()
        overlay[motion_masks[i] > 0.5] = [255, 0, 0]
        cv2.imwrite(str(output_dir / f"mask_{i:03d}.png"), 
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    # Extract tokens from all frames
    print(f"\n{'='*60}")
    print("EXTRACTING TOKENS")
    print(f"{'='*60}")
    
    all_tokens = []
    
    for i, frame in enumerate(frames):
        if (i + 1) % 5 == 0:
            print(f"  Frame {i+1}/{len(frames)}...")
        
        tokens_dict = extractor.extract_tokens_from_frames([frame])
        
        if i == 0:
            for layer_name in tokens_dict.keys():
                all_tokens.append({'layer': layer_name, 'tokens': []})
                print(f"  {layer_name}: {tokens_dict[layer_name].shape}")
        
        for j, layer_name in enumerate(tokens_dict.keys()):
            all_tokens[j]['tokens'].append(tokens_dict[layer_name].clone())
        
        torch.cuda.empty_cache()
    
    # Spatial Analysis
    print(f"\n{'='*60}")
    print("SPATIAL ANALYSIS")
    print(f"{'='*60}")
    
    spatial_analyzer = SpatialAnalyzer(output_dir=str(output_dir / "spatial"))
    spatial_results = {}
    
    mid_idx = len(frames) // 2
    
    for layer_data in all_tokens:
        layer_name = layer_data['layer']
        frame_tokens = layer_data['tokens'][mid_idx]
        
        if frame_tokens.dim() == 3:
            frame_tokens = frame_tokens[0]
        
        print(f"\n{layer_name}:")
        stats = spatial_analyzer.analyze_token_spatial_structure(
            frame_tokens, frames[mid_idx], layer_name
        )
        
        if 'grid_shape' in stats:
            h, w = stats['grid_shape']
            semantic_stats = spatial_analyzer.compare_semantic_regions(
                frame_tokens, motion_masks[mid_idx], h, w, layer_name
            )
            stats['semantic_comparison'] = semantic_stats
        
        spatial_results[layer_name] = stats
    
    with open(output_dir / "spatial_results.json", 'w') as f:
        json.dump(spatial_results, f, indent=2)
    
    # Temporal Analysis
    print(f"\n{'='*60}")
    print("TEMPORAL ANALYSIS (DELTAS)")
    print(f"{'='*60}")
    
    temporal_results = {}
    
    for layer_data in all_tokens:
        layer_name = layer_data['layer']
        token_sequence = torch.stack(layer_data['tokens'])
        
        if token_sequence.dim() == 4:
            token_sequence = token_sequence[:, 0, :, :]
        
        print(f"\n{layer_name}:")
        print(f"  Sequence: {token_sequence.shape}")
        
        # Compute deltas
        deltas = token_sequence[1:] - token_sequence[:-1]
        delta_norms = torch.norm(deltas, dim=2).numpy()
        
        print(f"  Delta: mean={delta_norms.mean():.4f}, std={delta_norms.std():.4f}")
        
        # Motion vs Static comparison
        if layer_name in spatial_results and 'grid_shape' in spatial_results[layer_name]:
            h, w = spatial_results[layer_name]['grid_shape']
            
            motion_deltas = []
            static_deltas = []
            
            for t in range(len(delta_norms)):
                if t >= len(motion_masks):
                    break
                
                mask_resized = cv2.resize(
                    motion_masks[t].astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST
                ).flatten()
                
                frame_deltas = delta_norms[t]
                actual_tokens = min(len(frame_deltas), len(mask_resized))
                
                motion_deltas.extend(frame_deltas[:actual_tokens][mask_resized[:actual_tokens] > 0.5].tolist())
                static_deltas.extend(frame_deltas[:actual_tokens][mask_resized[:actual_tokens] <= 0.5].tolist())
            
            motion_deltas = np.array(motion_deltas)
            static_deltas = np.array(static_deltas)
            
            if len(motion_deltas) > 0 and len(static_deltas) > 0:
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(motion_deltas, static_deltas)
                
                print(f"  Motion: {motion_deltas.mean():.4f} ± {motion_deltas.std():.4f}")
                print(f"  Static: {static_deltas.mean():.4f} ± {static_deltas.std():.4f}")
                print(f"  Ratio: {motion_deltas.mean()/static_deltas.mean():.2f}x")
                print(f"  t-test: t={t_stat:.4f}, p={p_value:.2e}")
                
                if p_value < 0.001:
                    print(f"  *** DELTAS ARE MEANINGFUL! (p < 0.001)")
                
                temporal_results[layer_name] = {
                    'motion_mean': float(motion_deltas.mean()),
                    'static_mean': float(static_deltas.mean()),
                    'ratio': float(motion_deltas.mean() / static_deltas.mean()),
                    't_stat': float(t_stat),
                    'p_value': float(p_value),
                }
    
    with open(output_dir / "temporal_results.json", 'w') as f:
        json.dump(temporal_results, f, indent=2)
    
    # Summary
    print(f"\n{'='*80}")
    print("QWEN3-VL (SigLIP2) SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n1. SPATIAL STRUCTURE:")
    for layer_name, stats in spatial_results.items():
        if 'grid_shape' in stats:
            h, w = stats['grid_shape']
            print(f"   {layer_name}: {h}×{w} = {h*w} tokens")
    
    print(f"\n2. SEMANTIC DIFFERENTIATION:")
    for layer_name, stats in spatial_results.items():
        if 'semantic_comparison' in stats and stats['semantic_comparison']:
            ratio = stats['semantic_comparison'].get('ratio', 0)
            p_val = stats['semantic_comparison'].get('p_value', 1)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.05 else "ns"
            print(f"   {layer_name}: Object/Background = {ratio:.2f}x {sig}")
    
    print(f"\n3. MOTION ENCODING:")
    for layer_name, stats in temporal_results.items():
        ratio = stats.get('ratio', 0)
        p_val = stats.get('p_value', 1)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.05 else "ns"
        print(f"   {layer_name}: Motion/Static = {ratio:.2f}x {sig}")
    
    print(f"\n{'='*80}")
    print(f"COMPARISON WITH QWEN2-VL")
    print(f"{'='*80}")
    
    print(f"\nQwen2-VL Results:")
    print(f"  visual.blocks.last: 1.14x ratio (p < 1e-13)")
    print(f"  visual.merger:      1.28x ratio (p < 1e-48)")
    
    print(f"\nQwen3-VL Results (SigLIP2):")
    for layer_name, stats in temporal_results.items():
        ratio = stats.get('ratio', 0)
        p_val = stats.get('p_value', 1)
        print(f"  {layer_name}: {ratio:.2f}x ratio (p = {p_val:.2e})")
    
    print(f"\nAll results saved to: {output_dir}/")
    
    extractor.cleanup()
    del extractor
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n✅ Qwen3-VL analysis complete!")


if __name__ == "__main__":
    main()
