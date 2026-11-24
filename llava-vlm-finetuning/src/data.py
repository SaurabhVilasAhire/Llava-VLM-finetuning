"""Data utilities: JSON <-> Donut-style token-sequence conversion, and the
LLaVA fine-tuning dataset wrapper.

The token format encodes a JSON object as nested <s_key>...</s_key> tags,
e.g. {"menu": {"name": "Coffee"}} becomes
<s_menu><s_name>Coffee</s_name></s_menu>. This lets a vision-language model
emit structured JSON as a flat token stream that a standard causal LM head
can be trained to generate, and that can be parsed back into JSON at
inference time.
"""
from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Optional

try:
    from torch.utils.data import Dataset
except ImportError:  # torch isn't needed just to use json2token/token2json
    class Dataset:  # type: ignore[no-redef]
        pass

try:
    from datasets import load_dataset
except ImportError:  # only needed when actually instantiating LlavaDataset
    load_dataset = None


def json2token(obj: Any, sort_json_key: bool = True) -> str:
    """Flatten a (possibly nested) JSON object into a token sequence."""
    if isinstance(obj, dict):
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        keys = sorted(obj.keys(), reverse=True) if sort_json_key else list(obj.keys())
        return "".join(f"<s_{k}>{json2token(obj[k], sort_json_key)}</s_{k}>" for k in keys)
    if isinstance(obj, list):
        return "<sep/>".join(json2token(item, sort_json_key) for item in obj)
    return str(obj)


def token2json(tokens: str, added_vocab: Optional[set] = None, is_inner_value: bool = False):
    """Parse a generated token sequence back into JSON. Inverse of json2token,
    tolerant of a model's generation mistakes (e.g. an unmatched closing tag
    is simply dropped rather than raising)."""
    output: Dict[str, Any] = {}
    added_vocab = added_vocab or set()

    while tokens:
        start_match = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
        if start_match is None:
            break
        key = start_match.group(1)
        key_escaped = re.escape(key)
        start_token = start_match.group()

        end_match = re.search(rf"</s_{key_escaped}>", tokens, re.IGNORECASE)
        if end_match is None:
            tokens = tokens.replace(start_token, "")
            continue

        end_token = end_match.group()
        content_match = re.search(
            f"{re.escape(start_token)}(.*?){re.escape(end_token)}", tokens, re.IGNORECASE | re.DOTALL
        )
        if content_match is not None:
            content = content_match.group(1).strip()
            if "<s_" in content and "</s_" in content:  # nested object
                value = token2json(content, added_vocab, is_inner_value=True)
                if value:
                    output[key] = value[0] if len(value) == 1 else value
            else:  # leaf value(s)
                leaves: List[str] = []
                for leaf in content.split("<sep/>"):
                    leaf = leaf.strip()
                    if leaf in added_vocab and leaf.startswith("<") and leaf.endswith("/>"):
                        leaf = leaf[1:-2]
                    leaves.append(leaf)
                output[key] = leaves[0] if len(leaves) == 1 else leaves

        tokens = tokens[tokens.find(end_token) + len(end_token):].strip()
        if tokens[:6] == "<sep/>":
            return [output] + token2json(tokens[6:], added_vocab, is_inner_value=True)

    if output:
        return [output] if is_inner_value else output
    return [] if is_inner_value else {"text_sequence": tokens}


class LlavaDataset(Dataset):
    """HuggingFace-dataset wrapper that turns (image, ground_truth JSON) pairs
    from CORD-v2 into (image, token_sequence) pairs ready for LLaVA-NeXT
    fine-tuning."""

    def __init__(
        self,
        dataset_name_or_path: str,
        split: str = "train",
        sort_json_key: bool = True,
        max_samples: Optional[int] = 100,
    ):
        super().__init__()
        self.split = split
        self.sort_json_key = sort_json_key

        self.dataset = load_dataset(dataset_name_or_path, split=split)
        if max_samples is not None:
            self.dataset = self.dataset.take(max_samples)
        self.dataset_length = len(self.dataset)

        self.gt_token_sequences = []
        for sample in self.dataset:
            ground_truth = json.loads(sample["ground_truth"])
            if "gt_parses" in ground_truth:
                gt_jsons = ground_truth["gt_parses"]
            else:
                gt_jsons = [ground_truth["gt_parse"]]
            self.gt_token_sequences.append(
                [json2token(gt_json, self.sort_json_key) for gt_json in gt_jsons]
            )

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        image = sample["image"]
        target_sequence = random.choice(self.gt_token_sequences[idx])
        return image, target_sequence
