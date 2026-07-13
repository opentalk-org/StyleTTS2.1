from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from runner.nodes.assets.credentials import huggingface_token
from runner.nodes.tts.engines.base import EngineRuntime, EngineSynthesisRequest, EngineSynthesisResult
from runner.nodes.tts.voices import Voice

# Unsloth's ungated re-host of canopylabs/orpheus-3b-0.1-ft (same finetuned weights)
# so synthesis needs no gated-repo access.
ORPHEUS_REPO_ID = "unsloth/orpheus-3b-0.1-ft"
ORPHEUS_SAMPLE_RATE = 24000

# Prompt framing tokens from Orpheus (`voice: text` wrapped in start/end control ids).
_START_TOKEN = 128259
_END_TOKENS = (128009, 128260, 128261, 128257)
_STOP_TOKEN_ID = 49158


class OrpheusRuntime(EngineRuntime):
    """Orpheus (Llama-3.2-3B → SNAC codec), driven through vLLM's offline LLM API.

    The upstream ``OrpheusModel`` wrapper targets an old vLLM async API and has bugs
    (validate_voice references a missing attribute, stale voice list), so we call vLLM
    directly and reuse only Orpheus's SNAC token decoder.
    """

    SAMPLE_RATE = ORPHEUS_SAMPLE_RATE

    def __init__(self, llm: Any, tokenizer: Any, sampling_cls: Any, decode_tokens: Any):
        self._llm = llm
        self._tokenizer = tokenizer
        self._sampling_cls = sampling_cls
        self._decode_tokens = decode_tokens

    def synthesize(self, text: str, voice: Voice, language: str) -> tuple[np.ndarray, int]:
        result = self.synthesize_batch([EngineSynthesisRequest(text, voice, language)], lambda: None)[0]
        return result.samples, result.sample_rate

    def synthesize_batch(
        self,
        requests: list[EngineSynthesisRequest],
        check_cancel: Callable[[], None],
    ) -> list[EngineSynthesisResult]:
        check_cancel()
        prompts = [
            self._format_prompt(request.text, request.voice.require_preset())
            for request in requests
        ]
        params = self._sampling_cls(
            temperature=0.6, top_p=0.8, max_tokens=1200, stop_token_ids=[_STOP_TOKEN_ID], repetition_penalty=1.3
        )
        outputs = self._llm.generate(prompts, params)
        check_cancel()
        assert len(outputs) == len(requests), "orpheus batch output mismatch"
        return [self._decode_output(output) for output in outputs]

    def _decode_output(self, output: Any) -> EngineSynthesisResult:
        token_ids = list(output.outputs[0].token_ids)
        audio_bytes = b"".join(self._decode_tokens(self._token_text_stream(token_ids)))
        if not audio_bytes:
            raise RuntimeError("orpheus_empty_audio")
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return EngineSynthesisResult(samples, ORPHEUS_SAMPLE_RATE)

    def _format_prompt(self, text: str, voice: str) -> str:
        import torch

        prompt_ids = self._tokenizer(f"{voice}: {text}", return_tensors="pt").input_ids
        start = torch.tensor([[_START_TOKEN]], dtype=torch.int64)
        end = torch.tensor([list(_END_TOKENS)], dtype=torch.int64)
        all_ids = torch.cat([start, prompt_ids, end], dim=1)
        return self._tokenizer.decode(all_ids[0])

    def _token_text_stream(self, token_ids: list[int]):
        # Orpheus's decoder parses the "<custom_token_N>" text of each generated token.
        for token_id in token_ids:
            yield self._tokenizer.decode([token_id])


def load(checkpoint_dir: Path, device: str | None = None) -> OrpheusRuntime:
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        from orpheus_tts.decoder import tokens_decoder_sync
    except ImportError as exc:
        raise RuntimeError("orpheus_not_installed") from exc
    # The Orpheus weights are a gated HF repo; vLLM/transformers read the token from the
    # environment, so surface the stored HF token there before any download happens.
    token = huggingface_token()
    if token and "HF_TOKEN" not in os.environ:
        os.environ["HF_TOKEN"] = token
    # Run vLLM's v1 engine core in-process: embedded in a runner node there is no
    # __main__ guard, so the default spawn-based EngineCore would re-import and re-enter.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    # flashinfer's JIT sampling kernel links libcudart.so.13 (CUDA 13); on the cu128 stack
    # fall back to native sampling / FlashAttention so nothing needs the CUDA-13 runtime.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    llm = LLM(model=str(checkpoint_dir), dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.5)
    # The finetuned model bundles its own tokenizer; use it (the pretrained repo is also gated).
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    return OrpheusRuntime(llm, tokenizer, SamplingParams, tokens_decoder_sync)
