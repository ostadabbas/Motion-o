import torch
from typing import List, Tuple, Optional
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

# Import specific model classes for different Qwen VL versions
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import Qwen2VLForConditionalGeneration
except ImportError:
    Qwen2VLForConditionalGeneration = None


class VLM:
    def __init__(self,
                 model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
                 device: str = "cuda",
                 dtype: str = "auto",
                 load_4bit: bool = True,
                 cache_dir: Optional[str] = None,
                 merge_adapters: bool = False):
        """
        Vision-language model loader.

        Notes:
            - If `model_id` is a Hugging Face model ID (e.g. \"Qwen/Qwen2-VL-2B-Instruct\"),
              we load the full base model.
            - If `model_id` is a local directory containing LoRA/PEFT adapters
              (e.g. ./outputs/stage2), we:
                * load the base Qwen2-VL model, and
                * load adapters from `model_id` with `PeftModel.from_pretrained`.
        """
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.load_4bit = load_4bit
        self.cache_dir = cache_dir

        quant_config = None
        torch_dtype = None
        if dtype == "auto":
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        if load_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        model_path = Path(self.model_id)
        is_peft_adapter = model_path.is_dir() and (model_path / "adapter_config.json").exists()

        if is_peft_adapter:
            # We fine-tuned adapters on top of the base Qwen3-VL model.
            # Use the same base ID as training (FinetuneConfig default).
            base_model_id = "Qwen/Qwen3-VL-8B-Instruct"

            # Load processor from base model
            self.processor = AutoProcessor.from_pretrained(
                base_model_id,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
            )

            # Select model class based on base_model_id
            model_class = self._get_model_class(base_model_id)
            
            # Load base model
            base_model = model_class.from_pretrained(
                base_model_id,
                torch_dtype=torch_dtype,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=self.cache_dir,
            )

            # Load PEFT adapters
            self.model = PeftModel.from_pretrained(base_model, self.model_id)
            
            # Verify adapters are loaded and active
            print(f"✓ Loaded PEFT adapters from {self.model_id}")
            if hasattr(self.model, 'active_adapters'):
                print(f"  Active adapters: {self.model.active_adapters}")
            if hasattr(self.model, 'peft_config'):
                print(f"  Peft config keys: {list(self.model.peft_config.keys())}")
            
            # Optionally merge adapters for inference (can help with quantized models)
            if merge_adapters:
                print("  Merging adapters into base model...")
                self.model = self.model.merge_and_unload()
                print("  ✓ Merged adapters into base model")
        else:
            # Regular base model load (no adapters)
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
            )
            
            # Select model class
            model_class = self._get_model_class(self.model_id)
            
            self.model = model_class.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=self.cache_dir,
            )

        self.model.eval()

    def _get_model_class(self, model_id: str):
        """Get the appropriate model class for the given model ID."""
        if "Qwen3" in model_id or "qwen3" in model_id:
            if Qwen3VLForConditionalGeneration is None:
                raise ImportError("Qwen3VLForConditionalGeneration not available. Update transformers.")
            return Qwen3VLForConditionalGeneration
        elif "Qwen2.5" in model_id or "qwen2.5" in model_id or "Qwen2_5" in model_id:
            if Qwen2_5_VLForConditionalGeneration is None:
                raise ImportError("Qwen2_5_VLForConditionalGeneration not available. Update transformers.")
            return Qwen2_5_VLForConditionalGeneration
        else:
            # Default to Qwen2VL
            if Qwen2VLForConditionalGeneration is None:
                raise ImportError("Qwen2VLForConditionalGeneration not available. Update transformers.")
            return Qwen2VLForConditionalGeneration

    def build_messages(self,
                       system_prompt: str,
                       user_text: str,
                       images: List[Image.Image]):
        # Qwen2-VL expects multimodal messages
        content = []
        for im in images:
            content.append({"type": "image", "image": im})
        if user_text:
            content.append({"type": "text", "text": user_text})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        return messages

    @torch.no_grad()
    def generate(self,
                 frames: List[Tuple[Image.Image, float]],
                 system_prompt: Optional[str],
                 user_prompt: str,
                 max_new_tokens: int = 32,
                 temperature: float = 0.3,
                 top_p: float = 0.9) -> str:
        images = [f[0] for f in frames]
        messages = self.build_messages(system_prompt, user_prompt, images)

        # Build chat template text
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Prepare multimodal inputs (text + images)
        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
        )

        inputs = {k: (v.to(self.model.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
        )

        # Decode only newly generated tokens
        input_len = inputs["input_ids"].shape[-1]
        gen_only = output[:, input_len:]
        out_text = self.processor.batch_decode(gen_only, skip_special_tokens=True)
        return out_text[0] if out_text else ""
