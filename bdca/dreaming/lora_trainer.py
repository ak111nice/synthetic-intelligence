"""
Online LoRA Synaptic Plasticity Engine (Tononi SHY Regularized).
Attaches low-rank adapters (r=8) to Qwen2.5-0.5B on CPU,
trains on verified failure-repair pairs, and saves checkpoint adapters.
"""

import os
import time
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel


class PairDataset(Dataset):
    def __init__(self, pairs: List[Dict[str, str]], tokenizer, max_length: int = 384):
        self.samples = []
        for p in pairs:
            text = f"<|im_start|>user\n{p['prompt']}<|im_end|>\n<|im_start|>assistant\n```python\n{p['code']}\n```<|im_end|>"
            enc = tokenizer(text, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            # Mask out user prompt from loss calculation
            user_part = f"<|im_start|>user\n{p['prompt']}<|im_end|>\n<|im_start|>assistant\n"
            user_len = len(tokenizer(user_part)["input_ids"])
            labels[:user_len] = -100
            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class LoRASynapticTrainer:
    """
    Manages continuous parameter updates to the plastic LoRA buffer.
    Applies Tononi Synaptic Homeostasis Hypothesis (SHY) via weight decay downscaling.
    """

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        checkpoint_dir: str = "./checkpoints/active_lora",
        rank: int = 8,
        lr: float = 2e-4,
        tononi_decay: float = 0.0005,
    ):
        self.base_model_name = base_model_name
        self.checkpoint_dir = checkpoint_dir
        self.rank = rank
        self.lr = lr
        self.tononi_decay = tononi_decay

    def train_step(self, verified_pairs: List[Dict[str, str]], epochs: int = 2) -> Dict[str, Any]:
        """Train plastic LoRA parameters on newly assimilated pairs."""
        if not verified_pairs:
            return {"status": "SKIPPED", "loss": 0.0, "samples": 0}

        print(f"\n[LoRA Synaptic Consolidation] Training on {len(verified_pairs)} verified invariant repairs...")
        t0 = time.perf_counter()

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load base model on CPU with frozen backbone
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        for param in base_model.parameters():
            param.requires_grad = False

        # Load existing LoRA adapter or initialize fresh
        if os.path.exists(os.path.join(self.checkpoint_dir, "adapter_config.json")):
            print("  Resuming from existing LoRA adapter checkpoint...")
            model = PeftModel.from_pretrained(base_model, self.checkpoint_dir, is_trainable=True)
        else:
            print("  Initializing fresh LoRA plastic adapter (r=8)...")
            peft_config = LoraConfig(
                r=self.rank,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(base_model, peft_config)

        model.train()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable Plastic Parameters: {trainable_params} (Only ~0.2% of 8B/0.5B model!)")

        # Tononi Downscaling implemented via decoupled weight decay in AdamW
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr,
            weight_decay=self.tononi_decay,
        )

        dataset = PairDataset(verified_pairs, tokenizer)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        total_loss = 0.0
        step_count = 0

        for ep in range(epochs):
            for batch in loader:
                optimizer.zero_grad()
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                step_count += 1

        avg_loss = total_loss / max(1, step_count)
        
        # Save updated plastic adapter
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        model.save_pretrained(self.checkpoint_dir)
        tokenizer.save_pretrained(self.checkpoint_dir)

        elapsed = time.perf_counter() - t0
        print(f"  ✓ Plastic LoRA updated & saved to '{self.checkpoint_dir}' in {elapsed:.2f}s (Loss: {avg_loss:.4f})\n")

        return {
            "status": "CONSOLIDATED",
            "loss": avg_loss,
            "samples": len(verified_pairs),
            "elapsed_sec": elapsed,
        }

    def bake_and_promote_generation(self, generation: int, export_base_dir: str = "./models") -> str:
        """
        Merges plastic LoRA weights into the frozen base model,
        exporting a permanent, standalone Generation Checkpoint (Gen-N).
        Resets active LoRA adapter for subsequent evolutionary cycles.
        """
        import shutil
        gen_dir = os.path.join(export_base_dir, f"synthetic-0.5b-gen{generation}")
        print(f"\n[Generation Hopping] Merging LoRA into base model -> {gen_dir}...")
        t0 = time.perf_counter()

        # Load base model + active LoRA adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)

        peft_model = PeftModel.from_pretrained(base_model, self.checkpoint_dir)
        # Merge low-rank delta matrices permanently into base weights
        merged_model = peft_model.merge_and_unload()

        # Save as standalone new foundation model
        os.makedirs(gen_dir, exist_ok=True)
        merged_model.save_pretrained(gen_dir)
        tokenizer.save_pretrained(gen_dir)

        # Clear out previous LoRA checkpoint so next generation trains fresh on new base
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)

        elapsed = time.perf_counter() - t0
        print(f"  ✓ Generation {generation} successfully baked and exported to '{gen_dir}' in {elapsed:.2f}s!\n")
        
        # Point trainer's base model to the newly baked model for next rounds
        self.base_model_name = gen_dir
        return gen_dir
