"""Load HF causal LMs within the 8 GB VRAM budget.

One rule, applied everywhere (docs/parts-plan.md Part 4): models above
``FOURBIT_THRESHOLD_B`` parameters load 4-bit via bitsandbytes, everything else
loads fp16 on CUDA (fp32 on CPU). Non-negotiable #5 caps model size at 2B for
anything *trained*; generators are only ever run for inference, and 4-bit is
what keeps a 7-9B generator inside the VRAM floor.

Serves docs/parts-plan.md Part 4.
"""
from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)

#: Above this many billion parameters, load 4-bit instead of fp16.
FOURBIT_THRESHOLD_B = 3.0

_SIZE_IN_NAME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")

#: Parameter counts (billions) for models whose id does not state a size.
KNOWN_SIZES_B: dict[str, float] = {
    "ai-forever/mgpt": 1.3,
    "google/muril-base-cased": 0.24,
    "xlm-roberta-base": 0.28,
}


def estimate_params_b(model_id: str) -> float | None:
    """Best-effort parameter count in billions, or None if it cannot be determined.

    Tries, in order: a known-sizes table, a size token in the id ("7B", "0.5b"),
    then the Hub's safetensors metadata. The id is checked before the Hub so the
    common case needs no network call.
    """
    known = KNOWN_SIZES_B.get(model_id.lower())
    if known is not None:
        return known
    match = _SIZE_IN_NAME_RE.search(model_id)
    if match:
        return float(match.group(1))
    try:
        from huggingface_hub import model_info

        total = (model_info(model_id).safetensors or {}).total
        if total:
            return total / 1e9
    except Exception as exc:  # noqa: BLE001 - offline or gated; fall through to None
        log.debug("could not read parameter count for %s: %s", model_id, exc)
    return None


def should_use_4bit(model_id: str, threshold_b: float = FOURBIT_THRESHOLD_B) -> bool:
    """True if ``model_id`` is large enough to need 4-bit quantisation.

    An unknown size is treated as large: guessing fp16 for a 9B model means an
    out-of-memory crash hours into a run, while guessing 4-bit for a small one
    only costs a little quality.
    """
    params_b = estimate_params_b(model_id)
    if params_b is None:
        log.warning("unknown parameter count for %s — assuming >%.0fB and loading 4-bit",
                    model_id, threshold_b)
        return True
    return params_b > threshold_b


def load_causal_lm(model_id: str, device: str | None = None, threshold_b: float = FOURBIT_THRESHOLD_B):
    """Return ``(model, tokenizer, info)`` ready for inference.

    ``info`` records what was actually done (``device``, ``dtype``, ``params_b``)
    so a run log states its own precision rather than leaving it to be inferred.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    params_b = estimate_params_b(model_id)
    use_4bit = should_use_4bit(model_id, threshold_b) and device == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        # Causal LMs usually ship without a pad token; batching needs one, and
        # EOS is the conventional stand-in. Padding is masked out anyway.
        tokenizer.pad_token = tokenizer.eos_token
    # Generation must pad on the left or a batch's shorter prompts end up with
    # pad tokens between the prompt and the continuation.
    tokenizer.padding_side = "left"

    kwargs: dict[str, Any] = {}
    if use_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
        dtype = "4bit-nf4"
    else:
        kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32
        dtype = "fp16" if device == "cuda" else "fp32"

    log.info("loading %s (%s params) as %s on %s",
             model_id, f"{params_b:.1f}B" if params_b else "unknown", dtype, device)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not use_4bit:
        model = model.to(device)
    model.eval()
    return model, tokenizer, {"device": device, "dtype": dtype, "params_b": params_b}
