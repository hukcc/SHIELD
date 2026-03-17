"""Caption loading, look-up and pre-processing utilities."""

import json
import os

from .clip_utils import get_clip_text_features


def load_captions(caption_file):
    """Load captions from a JSONL file."""
    with open(os.path.expanduser(caption_file), "r") as f:
        return [json.loads(line) for line in f]


def find_text_by_image(image_name, captions):
    """Look up the caption associated with an image name."""
    for entry in captions:
        if entry["image"] == image_name:
            return entry["text"]
    return None


def process_caption(caption):
    """Convert a raw caption into model-input and attack variants.

    Returns (text_in, text_ad) where text_in ends with ". " (for embeddings)
    and text_ad ends with "." (for the CLIP attack).
    """
    processed = ".".join(caption.replace("\n\n", "").split(".")[:-1])
    text_in = processed + ". "
    text_ad = processed + "."
    return text_in, text_ad


def prepare_caption_inputs(
    image_file,
    captions,
    tokenizer,
    clip_model=None,
    clip_processor=None,
):
    """Prepare all caption-related inputs needed during generation.

    Returns (text_in, text_ad, input_cap_ids, cap_tensor).
    """
    caption = find_text_by_image(image_file, captions)
    text_in, text_ad = process_caption(caption)

    input_cap_ids = (
        tokenizer(text_in, return_tensors="pt")["input_ids"].cuda()
    )
    cap_tensor = get_clip_text_features(text_in, clip_model, clip_processor)

    return text_in, text_ad, input_cap_ids, cap_tensor
