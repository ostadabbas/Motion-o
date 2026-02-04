"""
Token Extractor for Qwen2-VL
Extracts visual tokens at various stages of the model architecture.
"""

import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import cv2
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


class TokenExtractor:
    """
    Extracts visual tokens from Qwen2-VL at different architectural stages.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct", device: str = "cuda"):
        """
        Initialize the token extractor with Qwen2-VL model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ('cuda' or 'cpu')
        """
        print(f"Loading {model_name}...")
        self.device = device
        
        # Load model with memory optimization
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 for efficiency
            device_map="auto",
            low_cpu_mem_usage=True,
            # Removed max_memory constraint - let it use available GPU memory
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        self.model.eval()
        
        # Storage for extracted tokens
        self.extracted_tokens = {}
        self.hooks = []
        
        print(f"Model loaded with memory optimization")
        print(f"Model architecture: {type(self.model)}")
        
    def _register_hooks(self):
        """
        Register forward hooks to capture intermediate activations.
        """
        self.extracted_tokens = {}
        
        # Remove old hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                # Store the output tensor
                if isinstance(output, tuple):
                    token_data = output[0].detach().cpu().float()  # Convert to float32
                else:
                    token_data = output.detach().cpu().float()  # Convert to float32
                
                # Clean any NaN or inf values
                token_data[torch.isnan(token_data)] = 0
                token_data[torch.isinf(token_data)] = 0
                
                self.extracted_tokens[name] = token_data
            return hook
        
        # Hook into visual encoder
        # Qwen2-VL architecture: visual -> vision_model
        if hasattr(self.model, 'visual'):
            # Vision transformer layers
            if hasattr(self.model.visual, 'blocks'):
                # Hook last block
                last_block_idx = len(self.model.visual.blocks) - 1
                hook = self.model.visual.blocks[last_block_idx].register_forward_hook(
                    make_hook("visual.blocks.last")
                )
                self.hooks.append(hook)
                print(f"Registered hook: visual.blocks.last (block {last_block_idx})")
                
            # Hook visual projection
            if hasattr(self.model.visual, 'merger'):
                hook = self.model.visual.merger.register_forward_hook(
                    make_hook("visual.merger")
                )
                self.hooks.append(hook)
                print("Registered hook: visual.merger")
        
        # Alternative architecture path
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_model'):
            vision_model = self.model.model.vision_model
            
            # Hook vision encoder output
            if hasattr(vision_model, 'encoder'):
                hook = vision_model.encoder.register_forward_hook(
                    make_hook("vision_model.encoder")
                )
                self.hooks.append(hook)
                print("Registered hook: vision_model.encoder")
            
            # Hook post-layernorm
            if hasattr(vision_model, 'post_layernorm'):
                hook = vision_model.post_layernorm.register_forward_hook(
                    make_hook("vision_model.post_layernorm")
                )
                self.hooks.append(hook)
                print("Registered hook: vision_model.post_layernorm")
        
        print(f"Total hooks registered: {len(self.hooks)}")
    
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
        
        # Prepare messages for Qwen2-VL
        # Use low resolution to save memory
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,  # Low resolution to save memory
                        "fps": 1.0,  # Extract 1 frame per second (8 frames for 8s video)
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
    
    # Test the extractor
    extractor = TokenExtractor()
    
    video_path = "test_videos/Ball_Animation_Video_Generation.mp4"
    
    # Load frames manually instead
    print("Loading frames manually...")
    frames, fps = extractor.load_video_frames(video_path, num_frames=8)
    
    # Extract tokens from frames
    tokens = extractor.extract_tokens_from_frames(frames[:1])  # Start with just 1 frame
    
    print("\n" + "="*60)
    print("EXTRACTED TOKENS INFO:")
    print("="*60)
    
    info = extractor.get_token_info()
    for layer_name, layer_info in info.items():
        print(f"\nLayer: {layer_name}")
        for key, value in layer_info.items():
            print(f"  {key}: {value}")
    
    extractor.cleanup()
    
    # Clean up memory
    del extractor
    gc.collect()
    torch.cuda.empty_cache()
