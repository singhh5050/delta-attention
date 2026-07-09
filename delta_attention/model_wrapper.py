import os
from typing import Dict, List, Optional

import requests
import torch
from .sample import init_model
from .config import Config
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class HuggingFaceModel:
    def __init__(self, config: Config, **generation_kwargs) -> None:
        model_kwargs = {"attn_implementation": config.attn_implementation}

        self.pipeline = None

        model, tokenizer = init_model(config)
        model.eval()
        self.model = model.cuda()

        # stores the original so that "sample" method can swap out during decoding
        self.model.config.attn_implementation_original = config.attn_implementation_original
        self.model.config._attn_implementation = config.attn_implementation
        self.model.config.mode = config.mode
        self.model.config.delta_lambda = config.delta_lambda
        self.model.config.sliding_window = config.sliding_window
        self.model.config.hip_attention_dense_layers = config.hip_attention_dense_layers
        self.model.config.hip_attention_last_dense = config.hip_attention_last_dense
        self.model.config.log_drift = config.log_drift
        self.model.config.stride_policy = config.stride_policy
        self.model.config.gamma_min = config.gamma_min
        self.model.config.gamma_max = config.gamma_max
        self.model.config.adapt_chunk = config.adapt_chunk
        self.model.config.adapt_threshold = config.adapt_threshold
        self.model.config.stride_trigger = config.stride_trigger
        self.model.config.adapt_k = config.adapt_k

        self.tokenizer = tokenizer

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop")

        if self.tokenizer.pad_token is None:
            # add pad token to allow batching (known issue for llama2)
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _flush_drift_stats(self) -> None:
        """Append per-layer drift summaries (set by delta_forward when
        log_drift is on) to the DELTA_DRIFT_LOG sidecar, then clear them."""
        drift_path = os.environ.get("DELTA_DRIFT_LOG")
        collected = []
        for m in self.model.modules():
            for attr in ("_drift_stats", "_stride_stats"):
                stats = getattr(m, attr, None)
                if stats is not None:
                    collected.append(stats)
                    setattr(m, attr, None)
        if not collected:
            return
        if not drift_path:
            return  # telemetry computed but no sink configured (direct server use)
        import json

        with open(drift_path, "a", encoding="utf-8") as f:
            for stats in sorted(collected, key=lambda s: s["layer"]):
                f.write(json.dumps(stats) + "\n")

    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch(prompt, **kwargs)[0]

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        # use default kwargs and overwirte with input kwargs from the server
        generation_kwargs = {**self.generation_kwargs, **kwargs}
        print(f"{self.generation_kwargs=}")
        print(f"{kwargs=}")

        if self.pipeline is None:
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            generated_ids = self.model.generate(**inputs, **generation_kwargs)
            generated_texts = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            self._flush_drift_stats()

        results = []

        for text, prompt in zip(generated_texts, prompts):
            # remove the input form the generated text
            # This is a workaround for the llama3 tokenizer not being able to reproduce the same prompt after tokenization
            # see Issue https://github.com/NVIDIA/RULER/issues/54 for explaination
            if self.pipeline is None:
                tokenized_prompt = self.tokenizer(
                    prompt, return_tensors="pt", padding=True
                )
                prompt = self.tokenizer.decode(
                    tokenized_prompt.input_ids[0], skip_special_tokens=True
                )

            if text.startswith(prompt):
                text = text[len(prompt) :]

            if self.stop is not None:
                for s in self.stop:
                    text = text.split(s)[0]

            results.append({"text": [text]})

        return results
