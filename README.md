# VLM Fine-Tuning for Structured Image-to-JSON Parsing

QLoRA fine-tuning of **LLaVA-NeXT (1.6-7B, Mistral backbone)** to turn document
images (receipts) into structured JSON, plus a Flask + Streamlit app to
demo the model interactively.

## How it works

1. **Task framing** — instead of a free-text VQA prompt, the model is trained
   to answer `Extract JSON` with a flattened, Donut-style token sequence:
   a JSON object like `{"menu": {"name": "Coffee"}}` becomes
   `<s_menu><s_name>Coffee</s_name></s_menu>`. This lets a standard causal LM
   head generate structured output token-by-token, which is then parsed back
   into JSON (`src/data.py::json2token` / `token2json`).
2. **Data** — [CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2),
   a receipt dataset with ground-truth key-value annotations.
3. **Fine-tuning** — QLoRA: the base model is loaded in 4-bit NF4
   (`bitsandbytes`), and a LoRA adapter (`r=8, alpha=8, dropout=0.1`) is
   applied to the language model's linear layers only — the vision tower and
   multimodal projector are left frozen (`src/model.py`). Training uses
   PyTorch Lightning with gradient accumulation, mixed precision, and early
   stopping on validation edit-distance (`src/train.py`).
4. **Serving** — a Flask endpoint loads the fine-tuned adapter and runs
   inference (`src/serve_model.py`); a Streamlit app uploads an image and
   renders the returned JSON (`src/app.py`).

## Repo layout

```
src/
  data.py          # JSON <-> token-sequence conversion, CORD-v2 dataset wrapper
  model.py          # QLoRA model loading (4-bit + LoRA)
  train.py          # training entrypoint (Lightning module + CLI)
  serve_model.py    # Flask inference server
  app.py            # Streamlit frontend
tests/
  test_data.py      # CPU-only unit tests for the tokenization logic
Notebooks/
  data_exploration/xplr_cord_v2_dataset.ipynb   # CORD-v2 structure/exploration
  test_models/test_llava-v1.6-mistral-7b-hf.ipynb  # sanity-check of the base model actually fine-tuned
data/
  image_0.png ... image_19.png   # sample CORD-v2 receipt images for the Streamlit demo
```

## Running it

```bash
pip install -r requirements.txt

# Train (requires a CUDA GPU, ~40-80GB class, for 4-bit inference + LoRA):
python src/train.py --epochs 1 --batch-size 1

# Serve + demo (also GPU-only):
python src/serve_model.py        # terminal 1
streamlit run src/app.py         # terminal 2
```

### What I could actually run here

The 4-bit quantized 7B model needs a CUDA GPU with ~40GB+ of memory for both
training and inference — not something available in my dev setup. What *is*
hardware-independent is the tokenization logic that the whole pipeline
depends on: converting ground-truth JSON into the model's training target,
and parsing the model's generated tokens back into JSON. I pulled that into
its own module (`src/data.py`) and wrote unit tests for it (round-trips on
flat objects, nested objects, and lists; and the case where the model
generates a malformed/unmatched tag):

```bash
python -m pytest tests/test_data.py -v
```

All 6 pass. The training and serving scripts are otherwise implemented and
ready to run, but unverified end-to-end on real GPU hardware by me.

## Credits

Adapted from [Farzad-R/Finetune-LLAVA-NEXT](https://github.com/Farzad-R/Finetune-LLAVA-NEXT)
(restructured from notebooks into modular scripts, with the parsing logic
pulled out, tested, and made import-safe without `torch`/`datasets`
installed).
