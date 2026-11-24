"""Model + PEFT (QLoRA) setup for fine-tuning LLaVA-NeXT."""
from typing import List, Sequence

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, LlavaNextForConditionalGeneration


def find_all_linear_names(
    model, skip_keywords: Sequence[str] = ("multi_modal_projector", "vision_model")
) -> List[str]:
    """Collect the names of linear layers to target with LoRA. The vision
    tower and the multimodal projector are skipped so only the language
    model's attention/MLP projections are adapted."""
    names = set()
    for name, module in model.named_modules():
        if any(keyword in name for keyword in skip_keywords):
            continue
        if isinstance(module, torch.nn.Linear):
            parts = name.split(".")
            names.add(parts[0] if len(parts) == 1 else parts[-1])
    names.discard("lm_head")  # keep the LM head in full precision
    return list(names)


def load_qlora_model(
    model_id: str,
    lora_r: int = 8,
    lora_alpha: int = 8,
    lora_dropout: float = 0.1,
):
    """Load the base model 4-bit NF4 quantized and wrap it with a LoRA
    adapter so fine-tuning only updates a small set of low-rank matrices."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, quantization_config=bnb_config,
    )
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    return model
