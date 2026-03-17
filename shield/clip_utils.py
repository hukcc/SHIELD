"""CLIP model loading, caching, and text feature extraction."""

import torch

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

_clip_cache = {
    "model": None,
    "processor": None,
}


def load_clip_model(
    model_name="openai/clip-vit-large-patch14-336",
    cache_dir="checkpoints/",
    device="cuda",
):
    """Load the CLIP model and processor with caching."""
    global _clip_cache

    if _clip_cache["model"] is None:
        from transformers import CLIPModel, CLIPProcessor

        _clip_cache["model"] = CLIPModel.from_pretrained(
            model_name, cache_dir=cache_dir
        ).to(device)
        _clip_cache["processor"] = CLIPProcessor.from_pretrained(
            model_name, cache_dir=cache_dir
        )

    return _clip_cache["model"], _clip_cache["processor"]


def clear_clip_cache():
    """Clear the cached CLIP objects and release GPU memory."""
    global _clip_cache
    _clip_cache["model"] = None
    _clip_cache["processor"] = None
    torch.cuda.empty_cache()


def get_clip_text_features(text, clip_model=None, clip_processor=None, device="cuda"):
    """Compute CLIP text features (hidden states excluding the CLS token)."""
    if clip_model is None or clip_processor is None:
        clip_model, clip_processor = load_clip_model(device=device)

    cap_ids = clip_processor(text=[text], return_tensors="pt", padding=True)[
        "input_ids"
    ].to(device)
    cap_tensor = clip_model.text_model(cap_ids).last_hidden_state[:, 1:, :]

    return cap_tensor
