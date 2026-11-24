"""Fine-tune LLaVA-NeXT (1.6-7B, Mistral backbone) with QLoRA on CORD-v2 for
structured image-to-JSON extraction.

Requires a CUDA GPU with enough memory to hold the 4-bit base model plus
activations (roughly a single 40-80GB-class GPU at batch size 1). There is
no CPU fallback for the quantized base model, so this script only runs on
GPU hardware.

Usage:
    python train.py --epochs 1 --batch-size 1 \
        --push-to-hub your-hf-username/llava-v1.6-mistral-7b-cordv2
"""
import argparse
import re

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from nltk import edit_distance
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from data import LlavaDataset
from model import load_qlora_model

MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
DATASET_ID = "naver-clova-ix/cord-v2"


def build_collate_fns(processor, max_length: int):
    """Returns the train/eval collate functions, closed over the processor
    and max sequence length so they can be passed to a DataLoader."""

    def train_collate_fn(examples):
        images, texts = [], []
        for image, ground_truth in examples:
            images.append(image)
            texts.append(f"[INST] <image>\nExtract JSON [\\INST] {ground_truth}")
        batch = processor(
            text=texts, images=images, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["image_sizes"], labels

    def eval_collate_fn(examples):
        images, texts, answers = [], [], []
        for image, ground_truth in examples:
            images.append(image)
            texts.append("[INST] <image>\nExtract JSON [\\INST]")
            answers.append(ground_truth)
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["image_sizes"], answers

    return train_collate_fn, eval_collate_fn


class LlavaModelPLModule(L.LightningModule):
    """PyTorch Lightning wrapper handling the training/validation loop,
    optimizer and dataloaders for QLoRA fine-tuning of LLaVA-NeXT."""

    def __init__(self, config, processor, model, train_dataset, val_dataset, train_collate_fn, eval_collate_fn):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_collate_fn = train_collate_fn
        self.eval_collate_fn = eval_collate_fn
        self.batch_size = config["batch_size"]

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, image_sizes, labels = batch
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values, image_sizes=image_sizes, labels=labels,
        )
        self.log("train_loss", outputs.loss)
        return outputs.loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        input_ids, attention_mask, pixel_values, image_sizes, answers = batch
        generated_ids = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values, image_sizes=image_sizes,
            max_new_tokens=self.config["max_length"],
        )
        predictions = self.processor.batch_decode(
            generated_ids[:, input_ids.size(1):], skip_special_tokens=True
        )

        scores = []
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))
        self.log("val_edit_distance", np.mean(scores))
        return scores

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config["lr"])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, collate_fn=self.train_collate_fn,
            batch_size=self.batch_size, shuffle=True, num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, collate_fn=self.eval_collate_fn,
            batch_size=self.batch_size, shuffle=False, num_workers=4,
        )


class PushToHubCallback(Callback):
    """Pushes the LoRA adapter to the Hugging Face Hub after every epoch,
    and the processor alongside the final adapter when training ends."""

    def __init__(self, repo_id: str):
        self.repo_id = repo_id

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.model.push_to_hub(
            self.repo_id, commit_message=f"Training in progress, epoch {trainer.current_epoch}"
        )

    def on_train_end(self, trainer, pl_module):
        pl_module.processor.push_to_hub(self.repo_id, commit_message="Training done")
        pl_module.model.push_to_hub(self.repo_id, commit_message="Training done")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulate-grad-batches", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--push-to-hub", default=None, help="HF Hub repo id to push checkpoints to")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "right"  # always pad on the right during training

    model = load_qlora_model(args.model_id)

    train_dataset = LlavaDataset(
        args.dataset_id, split="train", sort_json_key=False, max_samples=args.max_train_samples
    )
    val_dataset = LlavaDataset(
        args.dataset_id, split="validation", sort_json_key=False, max_samples=args.max_train_samples
    )
    train_collate_fn, eval_collate_fn = build_collate_fns(processor, args.max_length)

    config = {"batch_size": args.batch_size, "lr": args.lr, "max_length": args.max_length}
    module = LlavaModelPLModule(
        config, processor, model, train_dataset, val_dataset, train_collate_fn, eval_collate_fn
    )

    callbacks = [EarlyStopping(monitor="val_edit_distance", patience=3, mode="min")]
    if args.push_to_hub:
        callbacks.append(PushToHubCallback(args.push_to_hub))

    trainer = L.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        precision="16-mixed",
        limit_val_batches=5,
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )
    trainer.fit(module)


if __name__ == "__main__":
    main()
