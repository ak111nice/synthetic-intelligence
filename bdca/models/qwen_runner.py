"""
Real Model Adapter for BDCA (Qwen2.5-0.5B Runner).
Connects a real neural network to the Cerebral Engine and Spinal Sentinel.
"""

from typing import Dict, Any, Optional
import os
import time

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class Qwen05BRunner:
    """
    Adapter that runs Qwen2.5-0.5B locally on CPU/GPU.
    Enforces structured JSON extraction, freezes backbone weights,
    and interfaces with SpinalSentinel.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cpu",
        torch_dtype: Optional[Any] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.base_model = None
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    def load(self) -> None:
        """Download (if needed) and load model weights into RAM."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch & Transformers must be installed to load Qwen05BRunner.")

        print(f"Loading '{self.model_name}' on {self.device.upper()}...")
        start_t = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
            trust_remote_code=True,
        ).to(self.device)

        # Architectural Rule: Freeze Backbone! (requires_grad = False)
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.eval()
        self.model = self.base_model
        self.is_loaded = True

        # Attach existing LoRA adapter if present
        self.reload_adapter()

        elapsed = time.perf_counter() - start_t
        print(f"✓ Model loaded and backbone weights frozen in {elapsed:.2f}s.")

    def reload_adapter(self, lora_checkpoint_dir: str = "./checkpoints/active_lora") -> None:
        """Dynamically attach or reload evolved plastic LoRA weights."""
        if not self.is_loaded:
            return
        if os.path.exists(os.path.join(lora_checkpoint_dir, "adapter_config.json")):
            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.base_model, lora_checkpoint_dir)
                for param in self.model.parameters():
                    param.requires_grad = False
                self.model.eval()
                print(f"  ✓ Dynamic plastic LoRA adapter attached from '{lora_checkpoint_dir}'")
            except Exception as e:
                print(f"  ⚠️ Warning: Failed to reload LoRA adapter: {e}")
        else:
            # Revert to base model if LoRA was merged or deleted
            self.model = self.base_model
            self.model.eval()

    def switch_base_model(self, new_model_path: str) -> None:
        """Promote newly baked generation model as the active inference backbone."""
        print(f"\n[Cerebral Substrate Promotion] Upgrading base model to '{new_model_path}'...")
        self.model_name = new_model_path
        self.load()

    def generate_action(self, prompt: str, max_new_tokens: int = 128) -> str:
        """Run forward inference on CPU to generate an action proposal."""
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call .load() first.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                top_p=0.9,
                do_sample=False,  # Deterministic decoding for invariants
            )

        output_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return output_text.strip()
