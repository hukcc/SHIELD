"""SHIELD -- non-invasive visual contrastive decoding with feature enhancement.

Quick start::

    import shield

    model = shield.wrap(model, tokenizer, caption_file="captions.jsonl")

    shield_kw = model.shield_prepare(image, image_tensor, image_file)
    output = model.generate(input_ids, **shield_kw, do_sample=True, max_new_tokens=1024)
"""

from .wrapper import wrap, DEFAULT_PARAMS

from .attack import cw_attack, perform_clip_attack, pgd_attack
from .caption import (
    find_text_by_image,
    load_captions,
    prepare_caption_inputs,
    process_caption,
)
from .clip_utils import clear_clip_cache, get_clip_text_features, load_clip_model
from .feature import (
    clear_bias_cache,
    compute_shield_image_weights,
    get_bias,
    map_text_segments,
    maximize_weight_difference,
)
from .noise import add_diffusion_noise
from .sampling import enable_shield_sampling

__all__ = [
    "wrap",
    "DEFAULT_PARAMS",
    "cw_attack",
    "pgd_attack",
    "perform_clip_attack",
    "load_clip_model",
    "clear_clip_cache",
    "get_clip_text_features",
    "load_captions",
    "find_text_by_image",
    "process_caption",
    "prepare_caption_inputs",
    "compute_shield_image_weights",
    "maximize_weight_difference",
    "map_text_segments",
    "get_bias",
    "clear_bias_cache",
    "add_diffusion_noise",
    "enable_shield_sampling",
]
