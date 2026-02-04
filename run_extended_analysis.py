"""
Extended Analysis with Full GPU Memory
Runs motion content tests and augmentation experiments.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import json
import gc

from token_extractor import TokenExtractor
from spatial_analysis import create_ball_mask
from motion_content_tests import MotionContentAnalyzer
from augmentation_test import AugmentationTester


def main():
    print("="*80)
    print("EXTENDED MOTION ANALYSIS")
    print("Testing motion information content and augmentation compatibility")
    print("="*80)
    
    # Configuration - use more frames with available GPU memory
    video_path = "test_videos/Ball_Animation_Video_Generation.mp4"
    num_frames = 30  # More frames for better analysis
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize
    extractor = TokenExtractor()
    
    # ========================================================================
    # LOAD FRAMES AND EXTRACT TOKENS
    # ========================================================================
    print(f"\n{'='*60}")
    print("LOADING FRAMES AND EXTRACTING TOKENS")
    print(f"{'='*60}")
    
    frames, fps = extractor.load_video_frames(video_path, num_frames=num_frames)
    print(f"Loaded {len(frames)} frames")
    
    # Create motion masks
    print("Creating motion masks...")
    motion_masks = [create_ball_mask(frame, method='color') for frame in frames]
    
    # Extract tokens
    print(f"Extracting tokens from {len(frames)} frames...")
    all_tokens = []
    
    for i, frame in enumerate(frames):
        if (i + 1) % 5 == 0:
            print(f"  Processing frame {i+1}/{len(frames)}...")
        
        tokens_dict = extractor.extract_tokens_from_frames([frame])
        
        if i == 0:
            for layer_name in tokens_dict.keys():
                all_tokens.append({'layer': layer_name, 'tokens': []})
        
        for j, layer_name in enumerate(tokens_dict.keys()):
            all_tokens[j]['tokens'].append(tokens_dict[layer_name].clone())
        
        torch.cuda.empty_cache()
    
    print(f"Token extraction complete!")
    
    # Get grid dimensions from spatial results
    with open(output_dir / "spatial_results.json", 'r') as f:
        spatial_results = json.load(f)
    
    # ========================================================================
    # MOTION CONTENT TESTS
    # ========================================================================
    print(f"\n{'='*60}")
    print("MOTION CONTENT TESTS")
    print(f"{'='*60}")
    
    motion_analyzer = MotionContentAnalyzer()
    motion_results = {}
    
    for layer_data in all_tokens:
        layer_name = layer_data['layer']
        token_sequence = torch.stack(layer_data['tokens'])
        
        if token_sequence.dim() == 4:
            token_sequence = token_sequence[:, 0, :, :]
        
        print(f"\n--- Layer: {layer_name} ---")
        
        # Get grid dimensions
        if layer_name in spatial_results and 'grid_shape' in spatial_results[layer_name]:
            h, w = spatial_results[layer_name]['grid_shape']
        else:
            continue
        
        # Compute deltas
        deltas = token_sequence[1:] - token_sequence[:-1]
        
        # Test 1: Predictive Power
        print("\n1. Testing predictive power...")
        pred_results = motion_analyzer.test_predictive_power(
            deltas, motion_masks, h, w, layer_name
        )
        
        # Test 2: Temporal Scaling
        print("\n2. Testing temporal scaling...")
        scaling_results = motion_analyzer.test_temporal_scaling(
            token_sequence, k_values=[1, 2, 3, 5, 10], layer_name=layer_name
        )
        
        # Test 3: Directional Information
        print("\n3. Testing directional information...")
        direction_results = motion_analyzer.test_directional_information(
            deltas, motion_masks, h, w, motion_direction="horizontal", layer_name=layer_name
        )
        
        # Test 4: Semantic Clustering
        print("\n4. Testing semantic clustering...")
        clustering_results = motion_analyzer.test_semantic_clustering(
            deltas, motion_masks, h, w, n_clusters=3, layer_name=layer_name
        )
        
        motion_results[layer_name] = {
            'predictive_power': pred_results,
            'temporal_scaling': scaling_results,
            'directional_info': direction_results,
            'clustering': clustering_results,
        }
    
    # Save motion content results
    with open(output_dir / "motion_content_results.json", 'w') as f:
        json.dump(motion_results, f, indent=2)
    
    print(f"\nMotion content analysis saved to {output_dir}/motion_content_results.json")
    
    # ========================================================================
    # AUGMENTATION TESTS
    # ========================================================================
    print(f"\n{'='*60}")
    print("AUGMENTATION COMPATIBILITY TESTS")
    print(f"{'='*60}")
    
    aug_tester = AugmentationTester(extractor.model, extractor.processor)
    augmentation_results = {}
    
    # Test baseline response
    print("\nTesting baseline model response...")
    baseline_result = aug_tester.test_baseline_response(video_path, max_frames=8)
    augmentation_results['baseline'] = baseline_result
    
    # Test augmented tokens
    print("\nTesting token augmentation...")
    for layer_data in all_tokens:
        layer_name = layer_data['layer']
        token_sequence = torch.stack(layer_data['tokens'])
        
        if token_sequence.dim() == 4:
            token_sequence = token_sequence[:, 0, :, :]
        
        print(f"\nLayer: {layer_name}")
        
        deltas = token_sequence[1:] - token_sequence[:-1]
        
        # Test different alpha values
        aug_results = aug_tester.test_augmented_tokens(
            token_sequence, deltas, alpha_values=[0.1, 0.5, 1.0, 2.0]
        )
        
        # Test noise tolerance
        noise_results = aug_tester.test_noise_tolerance(
            token_sequence, noise_levels=[0.01, 0.05, 0.1]
        )
        
        # Compare distributions
        if len(aug_results['alpha_tests']) > 0:
            alpha = 0.5
            augmented = token_sequence.clone()
            for t in range(min(len(deltas), len(augmented) - 1)):
                augmented[t + 1] = token_sequence[t + 1] + alpha * deltas[t]
            
            dist_comparison = aug_tester.compare_token_distributions(
                token_sequence, augmented
            )
            
            augmentation_results[layer_name] = {
                'alpha_tests': aug_results,
                'noise_tests': noise_results,
                'distribution_comparison': dist_comparison,
            }
    
    # Save augmentation results
    aug_tester.save_results(augmentation_results, "augmentation_results.json")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'='*80}")
    print("EXTENDED ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    print("\nKey Findings:")
    
    print("\n1. PREDICTIVE POWER:")
    for layer_name, results in motion_results.items():
        if 'predictive_power' in results and results['predictive_power']:
            best_f1 = results['predictive_power'].get('best_f1', 0)
            print(f"   {layer_name}: Best F1 = {best_f1:.3f}")
    
    print("\n2. TEMPORAL SCALING:")
    for layer_name, results in motion_results.items():
        if 'temporal_scaling' in results and results['temporal_scaling']:
            r2 = results['temporal_scaling'].get('r_squared', 0)
            print(f"   {layer_name}: R² = {r2:.3f} (linearity)")
    
    print("\n3. DIRECTIONAL INFORMATION:")
    for layer_name, results in motion_results.items():
        if 'directional_info' in results and results['directional_info']:
            pc1_var = results['directional_info'].get('pc1_variance', 0)
            print(f"   {layer_name}: PC1 explains {100*pc1_var:.1f}% variance")
    
    print("\n4. BASELINE MOTION AWARENESS:")
    if 'baseline' in augmentation_results:
        mentions = augmentation_results['baseline'].get('mentions_motion', False)
        print(f"   Model mentions motion: {mentions}")
        print(f"   Response: {augmentation_results['baseline'].get('response', '')[:100]}...")
    
    print(f"\nAll results saved to: {output_dir}/")
    
    # Cleanup
    extractor.cleanup()
    del extractor
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
