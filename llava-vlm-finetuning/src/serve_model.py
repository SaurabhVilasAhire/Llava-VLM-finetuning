"""Flask inference server: loads the fine-tuned LLaVA-NeXT model and exposes
a /process_image endpoint that returns structured JSON extracted from an
uploaded image. Requires a CUDA GPU to run (4-bit quantized inference)."""
import base64
import io

import torch
from flask import Flask, jsonify, request
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextForConditionalGeneration

from data import token2json

app = Flask(__name__)

MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
# Set this to your own fine-tuned adapter repo on the HF Hub after training.
FINE_TUNED_REPO_ID = "<your-hf-username>/llava-v1.6-mistral-7b-cordv2"
MAX_NEW_TOKENS = 256

processor = AutoProcessor.from_pretrained(MODEL_ID)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)
model = LlavaNextForConditionalGeneration.from_pretrained(
    FINE_TUNED_REPO_ID,
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
)


@app.route("/process_image", methods=["POST"])
def process_image():
    data = request.get_json()
    image_bytes = base64.b64decode(data["image"])
    image = Image.open(io.BytesIO(image_bytes))

    prompt = "[INST] <image>\nExtract JSON [/INST]"
    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
    output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    response = processor.decode(output[0], skip_special_tokens=True)

    generated_json = token2json(response, added_vocab=processor.tokenizer.get_added_vocab())
    return jsonify(generated_json)


if __name__ == "__main__":
    app.run(port=5000, debug=False)
