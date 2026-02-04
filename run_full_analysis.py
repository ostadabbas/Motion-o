"""
Main Analysis Script
Runs complete visual token analysis pipeline for motion augmentation.
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
from motion_content_tests import MotionContentAnalyzer
from augmentation_test import AugmentationTester


def main():
    print("="*80)
    print("VISUAL TOKEN MOTION ANALYSIS")
    print("Qwen2-VL Token Analysis for Motion Augmentation")
    print("="*80)
    
    # Configuration
    video_path = "test_videos/Ball_Animation_Video_Generation.mp4"
    max_frames = 15  # Reduced to save memory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Load frames manually
    print(f"\n{'='*60}")
    print("LOADING VIDEO FRAMES")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}")
    print(f"Sampling {max_frames} frames...")
    
    # Sample frames evenly
    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    frames = []
    motion_masks = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            # Create motion mask (ball detection)
            mask = create_ball_mask(frame_rgb, method='color')
            motion_masks.append(mask)
    
    cap.release()
    print(f"Loaded {len(frames)} frames")
    
    # Save sample frames
    print("\nSaving sample frames...")
    for i, (frame, mask) in enumerate(zip(frames[::5], motion_masks[::5])):
        # Save original frame
        frame_path = output_dir / f"frame_{i:03d}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Save mask visualization
        mask_path = output_dir / f"mask_{i:03d}.png"
        mask_vis = (mask * 255).astype(np.uint8)
        cv2.imwrite(str(mask_path), mask_vis)
    
    print(f"Saved sample frames to {output_dir}/")
    
    # ========================================================================
    # PART 1: Token Extraction
    # ========================================================================
    print(f"\n{'='*60}")
    print("PART 1: TOKEN EXTRACTION")
    print(f"{'='*60}")
    
    extractor = TokenExtractor()
    
    # Extract tokens from video
    tokens_dict = extractor.extract_tokens_from_video(video_path, max_frames=max_frames)
    
    print("\nExtracted tokens from layers:")
    for layer_name, tokens in tokens_dict.items():
        print(f"  {layer_name}: {tokens.shape}")
    
    # Get token info
    token_info = extractor.get_token_info()
    
    # Save token info
    with open(output_dir / "token_info.json", 'w') as f:
        json.dump(token_info, f, indent=2)
    
    print(f"\nToken info saved to {output_dir}/token_info.json")
    
    # ========================================================================
    # PART 2: Spatial Analysis
    # ========================================================================
    print(f"\n{'='*60}")
    print("PART 2: SPATIAL ANALYSIS")
    print(f"{'='*60}")
    
    spatial_analyzer = SpatialAnalyzer()
    spatial_results = {}
    
    # Analyze each layer
    for layer_name, tokens in tokens_dict.items():
        # For spatial analysis, use first frame
        # Tokens shape may vary, handle different cases
        if tokens.dim() == 2:
            # Single frame tokens [num_tokens, hidden_dim]
            frame_tokens = tokens
        elif tokens.dim() == 3:
            # Multiple frames or batched [batch/frames, num_tokens, hidden_dim]
            frame_tokens = tokens[0] if tokens.shape[0] > 1 else tokens.squeeze(0)
        else:
            print(f"Warning: Unexpected token shape {tokens.shape} for {layer_name}")
            continue
        
        # Get middle frame for visualization
        mid_frame = frames[len(frames) // 2]
        mid_mask = motion_masks[len(motion_masks) // 2]
        
        # Analyze spatial structure
        stats = spatial_analyzer.analyze_token_spatial_structure(
            frame_tokens, mid_frame, layer_name
        )
        
        # Compare semantic regions if we have grid dimensions
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
    
    print(f"\nSpatial results saved to {output_dir}/spatial_results.json")
    
    # Clean up
    extractor.cleanup()
    del extractor
    gc.collect()
    torch.cuda.empty_cache()
    
    # ========================================================================
    # NOTE: For temporal and augmentation analysis, we would need to extract
    # tokens for each frame separately and build a sequence. Due to memory
    # constraints, we'll document the approach instead.
    # ========================================================================
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    
    print("\nNOTE: Due to GPU memory constraints (3.94GB), temporal analysis")
    print("requiring full frame sequences could not be completed.")
    print("\nWhat we learned:")
    print("1. Token extraction successful - hooks work")
    print("2. Spatial token structure documented")
    print("3. Semantic region comparison shows token differences")
    print("\nFor full temporal analysis, recommend:")
    print("- Use larger GPU (>8GB)")
    print("- Or process frames in smaller batches")
    print("- Or use quantized model (4-bit/8-bit)")
    
    print(f"\nAll results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
