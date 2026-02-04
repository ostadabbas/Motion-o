"""
Token Augmentation Tests
Tests if motion-augmented tokens can be passed to the LLM without breaking it.
"""

import torch
import numpy as np
from typing import Dict, List
from pathlib import Path
import json


class AugmentationTester:
    """
    Tests compatibility of motion-augmented tokens with the language model.
    """
    
    def __init__(self, model, processor, output_dir: str = "results/augmentation"):
        """
        Initialize augmentation tester.
        
        Args:
            model: Qwen2-VL model
            processor: Qwen2-VL processor
            output_dir: Output directory for results
        """
        self.model = model
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Augmentation test output directory: {self.output_dir}")
    
    def test_baseline_response(
        self,
        video_path: str,
        max_frames: int = 10,
        prompt: str = "Describe what is happening in this video."
    ) -> Dict:
        """
        Get baseline model response with original tokens.
        
        Args:
            video_path: Path to video
            max_frames: Number of frames
            prompt: Prompt for the model
            
        Returns:
            Response dictionary
        """
        print(f"\n{'='*60}")
        print(f"BASELINE RESPONSE TEST")
        print(f"{'='*60}")
        
        from qwen_vl_utils import process_vision_info
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 256 * 256,
                        "fps": max_frames / 8.0,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to(self.model.device)
        
        # Generate response
        print("Generating baseline response...")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )
        
        # Decode
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(f"\nBaseline Response:")
        print(f"  {response}")
        
        # Check for motion-related words
        motion_keywords = ['move', 'moving', 'motion', 'left', 'right', 'across', 'ball']
        mentions_motion = any(keyword in response.lower() for keyword in motion_keywords)
        
        print(f"\nMentions motion: {mentions_motion}")
        if mentions_motion:
            print("  Model already has some motion awareness!")
        
        torch.cuda.empty_cache()
        
        return {
            'prompt': prompt,
            'response': response,
            'mentions_motion': mentions_motion,
            'response_length': len(response),
        }
    
    def test_augmented_tokens(
        self,
        token_sequence: torch.Tensor,
        deltas: torch.Tensor,
        alpha_values: List[float] = [0.1, 0.5, 1.0, 2.0]
    ) -> Dict:
        """
        Test augmented tokens with different alpha values.
        
        Args:
            token_sequence: Original tokens [num_frames, num_tokens, hidden_dim]
            deltas: Token deltas [num_frames-1, num_tokens, hidden_dim]
            alpha_values: List of alpha values to test
            
        Returns:
            Analysis results
        """
        print(f"\n{'='*60}")
        print(f"TOKEN AUGMENTATION TEST")
        print(f"{'='*60}")
        
        results = []
        
        for alpha in alpha_values:
            print(f"\nTesting alpha = {alpha}")
            
            # Create augmented tokens for middle frames
            augmented_sequence = token_sequence.clone()
            
            for t in range(min(len(deltas), len(augmented_sequence) - 1)):
                augmented_sequence[t + 1] = token_sequence[t + 1] + alpha * deltas[t]
            
            # Compute distribution shift
            original_mean = token_sequence.mean().item()
            original_std = token_sequence.std().item()
            augmented_mean = augmented_sequence.mean().item()
            augmented_std = augmented_sequence.std().item()
            
            mean_shift = abs(augmented_mean - original_mean) / original_std
            std_shift = abs(augmented_std - original_std) / original_std
            
            print(f"  Original: mean={original_mean:.6f}, std={original_std:.6f}")
            print(f"  Augmented: mean={augmented_mean:.6f}, std={augmented_std:.6f}")
            print(f"  Mean shift: {mean_shift:.2f} std deviations")
            print(f"  Std shift: {100*std_shift:.1f}%")
            
            results.append({
                'alpha': alpha,
                'original_mean': original_mean,
                'original_std': original_std,
                'augmented_mean': augmented_mean,
                'augmented_std': augmented_std,
                'mean_shift_std': mean_shift,
                'std_shift_pct': std_shift,
            })
        
        return {'alpha_tests': results}
    
    def test_noise_tolerance(
        self,
        token_sequence: torch.Tensor,
        noise_levels: List[float] = [0.01, 0.05, 0.1]
    ) -> Dict:
        """
        Test model's tolerance to random noise in tokens.
        
        Args:
            token_sequence: Original tokens
            noise_levels: List of noise standard deviations
            
        Returns:
            Noise tolerance analysis
        """
        print(f"\n{'='*60}")
        print(f"NOISE TOLERANCE TEST")
        print(f"{'='*60}")
        
        results = []
        
        token_std = token_sequence.std().item()
        
        for noise_level in noise_levels:
            print(f"\nNoise level: {noise_level} (relative to token std)")
            
            # Add Gaussian noise
            noise = torch.randn_like(token_sequence) * (noise_level * token_std)
            noisy_sequence = token_sequence + noise
            
            noise_magnitude = noise.norm(dim=2).mean().item()
            
            print(f"  Mean noise magnitude: {noise_magnitude:.6f}")
            
            results.append({
                'noise_level': noise_level,
                'noise_magnitude': noise_magnitude,
            })
        
        return {'noise_tests': results}
    
    def compare_token_distributions(
        self,
        original_tokens: torch.Tensor,
        augmented_tokens: torch.Tensor
    ) -> Dict:
        """
        Compare distributions of original vs augmented tokens.
        
        Args:
            original_tokens: Original token sequence
            augmented_tokens: Augmented token sequence
            
        Returns:
            Distribution comparison metrics
        """
        print(f"\n{'='*60}")
        print(f"DISTRIBUTION COMPARISON")
        print(f"{'='*60}")
        
        from scipy import stats as scipy_stats
        
        # Flatten for comparison
        orig_flat = original_tokens.flatten().numpy()
        aug_flat = augmented_tokens.flatten().numpy()
        
        # Sample for KS test (too slow on full data)
        sample_size = min(10000, len(orig_flat))
        indices = np.random.choice(len(orig_flat), sample_size, replace=False)
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = scipy_stats.ks_2samp(orig_flat[indices], aug_flat[indices])
        
        print(f"KS test: statistic={ks_stat:.6f}, p-value={ks_pvalue:.6e}")
        
        if ks_pvalue < 0.001:
            print("  *** Distributions are significantly different")
        else:
            print("  Distributions are similar")
        
        # Compute moments
        orig_moments = {
            'mean': float(orig_flat.mean()),
            'std': float(orig_flat.std()),
            'skew': float(scipy_stats.skew(orig_flat[::100])),  # Subsample for speed
            'kurtosis': float(scipy_stats.kurtosis(orig_flat[::100])),
        }
        
        aug_moments = {
            'mean': float(aug_flat.mean()),
            'std': float(aug_flat.std()),
            'skew': float(scipy_stats.skew(aug_flat[::100])),
            'kurtosis': float(scipy_stats.kurtosis(aug_flat[::100])),
        }
        
        print(f"\nOriginal moments:")
        for key, value in orig_moments.items():
            print(f"  {key}: {value:.6f}")
        
        print(f"\nAugmented moments:")
        for key, value in aug_moments.items():
            print(f"  {key}: {value:.6f}")
        
        return {
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'original_moments': orig_moments,
            'augmented_moments': aug_moments,
        }
    
    def save_results(self, results: Dict, filename: str = "augmentation_results.json"):
        """Save augmentation test results to JSON."""
        save_path = self.output_dir / filename
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    print("Augmentation Test Module")
    print("Run from main analysis script")
