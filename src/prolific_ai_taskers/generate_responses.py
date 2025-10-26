"""
LLM response generation module.
"""

import datetime
import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ResponseGenerator:
    """Generate multiple LLM responses for prompts."""

    def __init__(self, model_name: str):
        """
        Initialize with a language model.

        Args:
            model_name: HuggingFace model identifier (e.g., "meta-llama/Llama-3.2-3B")
        """
        self.model_name = model_name
        self.device = self._detect_device()
        self.torch_dtype = torch.float16 if self.device in ["cuda", "mps"] else torch.float32

        # Disable progress bars
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["HF_HUB_DISABLE_TQDM"] = "1"

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device_map = "auto" if self.device == "cuda" else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=self.torch_dtype,
            device_map=device_map,
            token=os.environ.get("HF_TOKEN"),
        )

        if device_map is None:
            self.model.to(self.device)

        print(f"✅ Model loaded on {self.device}")

    def _detect_device(self):
        """Detect best available device."""
        if torch.cuda.is_available():
            print("✅ Using NVIDIA GPU (CUDA)")
            return "cuda"
        elif torch.backends.mps.is_available():
            print("✅ Using Apple Silicon GPU (Metal)")
            return "mps"
        else:
            print("⚠️ No GPU found — using CPU")
            return "cpu"

    def generate_one(self, prompt: str, temperature: float = 0.7,
                     max_new_tokens: int = 512, top_p: float = 0.9) -> str:
        """
        Generate a single response.

        Args:
            prompt: Input text
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter

        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_out = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=int(max_new_tokens),
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        return self.tokenizer.decode(gen_out[0], skip_special_tokens=True)

    def generate_completions(self, prompts: List[Dict], temperatures: List[float],
                           num_completions_per_prompt: int, max_new_tokens: int = 512,
                           top_p: float = 0.9, verbose: bool = True) -> pd.DataFrame:
        """
        Generate multiple completions for prompts.

        Args:
            prompts: List of dicts with 'prompt' key
            temperatures: List of temperatures to use
            num_completions_per_prompt: Number of completions per prompt per temperature
            max_new_tokens: Maximum tokens per completion
            top_p: Nucleus sampling parameter
            verbose: Print progress

        Returns:
            DataFrame with all completions and metadata
        """
        rows = []
        run_id = str(uuid.uuid4())

        for p_idx, prompt_obj in enumerate(prompts):
            prompt_id = f"p{p_idx+1:04d}"
            prompt = prompt_obj["prompt"]

            if verbose:
                print(f"\n📝 Prompt {p_idx+1}: {prompt}")

            for temp in temperatures:
                if verbose:
                    print(f"  🌡️ Temperature={temp}")

                for c_idx in range(num_completions_per_prompt):
                    if verbose:
                        print(f"    ▶️ Generating completion {c_idx+1}", end="")

                    t0 = time.time()
                    try:
                        completion_text = self.generate_one(
                            prompt=prompt,
                            temperature=temp,
                            max_new_tokens=max_new_tokens,
                            top_p=top_p,
                        )
                    except Exception as e:
                        completion_text = f"[ERROR] {e}"

                    duration_s = time.time() - t0
                    if verbose:
                        print(f" done in {duration_s:.2f}s")

                    rows.append({
                        "run_id": run_id,
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "model": self.model_name,
                        "temperature": temp,
                        "top_p": top_p,
                        "max_new_tokens": max_new_tokens,
                        "completion_index": c_idx,
                        "response": completion_text,
                        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
                        "latency_seconds": round(duration_s, 4),
                    })

        if verbose:
            print("\n✅ Generation complete.")

        return pd.DataFrame(rows)

    def save_completions(self, df: pd.DataFrame, output_dir: Path,
                        config: Dict, num_prompts: int):
        """
        Save completions and metadata.

        Args:
            df: DataFrame with completions
            output_dir: Directory to save files
            config: Configuration dict
            num_prompts: Number of prompts processed

        Returns:
            Tuple of (csv_path, metadata_path)
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save CSV
        csv_path = output_dir / "completions.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"Wrote CSV → {csv_path} ({len(df)} rows)")

        # Save metadata
        metadata = {
            "run_id": df["run_id"].iloc[0] if len(df) > 0 else str(uuid.uuid4()),
            "started_at": df["generated_at"].iloc[0] if len(df) > 0 else datetime.datetime.utcnow().isoformat() + "Z",
            "finished_at": datetime.datetime.utcnow().isoformat() + "Z",
            "config": config,
            "num_prompts": num_prompts,
            "total_completions": len(df),
        }

        metadata_path = output_dir / "run_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Wrote metadata → {metadata_path}")
        return csv_path, metadata_path


def load_prompts(prompts_path: Path) -> List[Dict]:
    """Load prompts from JSONL file."""
    with open(prompts_path, "r") as f:
        return [json.loads(line) for line in f]
