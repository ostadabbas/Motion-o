"""
Token Extractor for Qwen3-VL (with SigLIP2 Vision Encoder)
Extracts visual tokens at various stages of the Qwen3-VL architecture.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
from qwen_vl_utils import process_vision_info
import cv2
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


class TokenExtractorQwen3:
    """
    Extracts visual tokens from Qwen3-VL (with SigLIP2 encoder) at different architectural stages.
    
    Key Difference from Qwen2-VL:
    - Vision Encoder: SigLIP2 (ViT-B/16 or larger) instead of Qwen's own ViT
    - Better multilingual vision-language alignment
    - Attention MAP pooling (4× token reduction)
    - Enhanced zero-shot capabilities
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct", device: str = "cuda"):
        """
        Initialize the token extractor with Qwen3-VL model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ('cuda' or 'cpu')
        """
        print(f"Loading {model_name}...")
        print("  Vision Encoder: SigLIP2 (multilingual ViT)")
        self.device = device
        
        # Load model - optimized for 32GB GPU
        # Use AutoModel to automatically load the correct model class
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 for efficiency
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,  # Required for Qwen3-VL
            # No max_memory constraint - use full GPU
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        self.model.eval()
        
        # Storage for extracted tokens
        self.extracted_tokens = {}
        self.hooks = []
        
        print(f"Model loaded successfully")
        print(f"Model architecture: {type(self.model)}")
        print(f"Vision model: {type(self.model.visual)}")
        
    def _register_hooks(self):
        """
        Register forward hooks to capture intermediate activations.
        
        For Qwen3-VL with SigLIP2:
        - SigLIP2 has different architecture than Qwen2's ViT
        - May have attention pooling layer
        - Merger layer for vision-language alignment
        """
        self.extracted_tokens = {}
        
        # Remove old hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                # Handle tuple output
                if isinstance(output, tuple):
                    token_data = output[0].detach().cpu().float()
                else:
                    token_data = output.detach().cpu().float()
                
                # Clean NaN/inf values
                token_data[torch.isnan(token_data)] = 0
                token_data[torch.isinf(token_data)] = 0
                
                self.extracted_tokens[name] = token_data
            return hook
        
        # Hook into SigLIP2 vision encoder
        if hasattr(self.model, 'visual'):
            # SigLIP2 Vision Transformer blocks
            if hasattr(self.model.visual, 'blocks'):
                # Hook last block (after all ViT layers)
                last_block_idx = len(self.model.visual.blocks) - 1
                hook = self.model.visual.blocks[last_block_idx].register_forward_hook(
                    make_hook("siglip2.blocks.last")
                )
                self.hooks.append(hook)
                print(f"Registered hook: siglip2.blocks.last (block {last_block_idx})")
            
            # Hook attention pooling layer (if exists - SigLIP2 feature)
            if hasattr(self.model.visual, 'attn_pool'):
                hook = self.model.visual.attn_pool.register_forward_hook(
                    make_hook("siglip2.attn_pool")
                )
                self.hooks.append(hook)
                print("Registered hook: siglip2.attn_pool")
            
            # Hook visual merger (vision-to-language projection)
            if hasattr(self.model.visual, 'merger'):
                hook = self.model.visual.merger.register_forward_hook(
                    make_hook("visual.merger")
                )
                self.hooks.append(hook)
                print("Registered hook: visual.merger")
        
        print(f"Total hooks registered: {len(self.hooks)}")
        
        if len(self.hooks) == 0:
            print("WARNING: No hooks registered! Architecture may have changed.")
            print("Model visual structure:")
            print(self.model.visual)
    
    def extract_tokens_from_video(
        self, 
        video_path: str, 
        max_frames: int = 30,
        prompt: str = "Describe what is happening in this video."
    ) -> Dict[str, torch.Tensor]:
        """
        Extract visual tokens from a video.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process
            prompt: Text prompt for the model
            
        Returns:
            Dictionary mapping layer names to token tensors
        """
        print(f"\nProcessing video: {video_path}")
        print(f"Max frames: {max_frames}")
        
        # Register hooks before processing
        self._register_hooks()
        
        # Prepare messages for Qwen3-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 720 * 1280,  # Higher resolution for 32GB GPU
                        "fps": 1.0,  # Extract evenly spaced frames
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
        
        inputs = inputs.to(self.device)
        
        print(f"Input shapes:")
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        
        # Forward pass to trigger hooks
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Clear cache
        torch.cuda.empty_cache()
        
        return self.extracted_tokens
    
    def extract_tokens_from_frames(
        self,
        frames: List[np.ndarray],
        prompt: str = "Describe what is happening."
    ) -> Dict[str, torch.Tensor]:
        """
        Extract tokens from a list of frame arrays.
        
        Args:
            frames: List of numpy arrays (H, W, C) in RGB format
            prompt: Text prompt
            
        Returns:
            Dictionary mapping layer names to token tensors
        """
        print(f"\nProcessing {len(frames)} frames directly")
        
        # Register hooks
        self._register_hooks()
        
        # Convert frames to PIL Images
        from PIL import Image
        pil_frames = [Image.fromarray(frame) for frame in frames]
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame}
                    for frame in pil_frames
                ] + [{"type": "text", "text": prompt}],
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
        
        inputs = inputs.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        torch.cuda.empty_cache()
        
        return self.extracted_tokens
    
    def get_token_info(self) -> Dict[str, Dict]:
        """
        Get information about extracted tokens.
        
        Returns:
            Dictionary with token statistics for each layer
        """
        info = {}
        for layer_name, tokens in self.extracted_tokens.items():
            info[layer_name] = {
                'shape': list(tokens.shape),
                'dtype': str(tokens.dtype),
                'device': str(tokens.device),
                'mean': float(tokens.mean()),
                'std': float(tokens.std()),
                'min': float(tokens.min()),
                'max': float(tokens.max()),
            }
        return info
    
    def load_video_frames(
        self, 
        video_path: str, 
        num_frames: int = 30
    ) -> Tuple[List[np.ndarray], float]:
        """
        Load evenly spaced frames from a video.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            
        Returns:
            Tuple of (list of frames as numpy arrays, fps)
        """
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video info: {total_frames} frames @ {fps} fps")
        
        # Sample evenly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        print(f"Loaded {len(frames)} frames")
        return frames, fps
    
    def cleanup(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


if __name__ == "__main__":
    import gc
    
    # Test the extractor with Qwen3-VL (SigLIP2)
    print("="*80)
    print("QWEN3-VL TOKEN EXTRACTOR (SigLIP2 Vision Encoder)")
    print("="*80)
    
    extractor = TokenExtractorQwen3()
    
    video_path = "test_videos/Ball_Animation_Video_Generation.mp4"
    
    # Load frames manually
    print("\nLoading frames manually...")
    frames, fps = extractor.load_video_frames(video_path, num_frames=8)
    
    # Extract tokens from first frame
    tokens = extractor.extract_tokens_from_frames(frames[:1])
    
    print("\n" + "="*60)
    print("EXTRACTED TOKENS INFO (SigLIP2):")
    print("="*60)
    
    info = extractor.get_token_info()
    for layer_name, layer_info in info.items():
        print(f"\nLayer: {layer_name}")
        for key, value in layer_info.items():
            print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("COMPARISON: Qwen2-VL vs Qwen3-VL")
    print("="*60)
    print("\nQwen2-VL (Custom ViT):")
    print("  - visual.blocks.last: [4784, 1280]")
    print("  - visual.merger: [1196, 3584]")
    print("\nQwen3-VL (SigLIP2):")
    for layer_name, layer_info in info.items():
        print(f"  - {layer_name}: {layer_info['shape']}")
    
    extractor.cleanup()
    
    # Clean up memory
    del extractor
    gc.collect()
    torch.cuda.empty_cache()
