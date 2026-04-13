import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import data_utils as du

# =========================
# CONFIG
# =========================

BASE_MODEL = "microsoft/phi-2"

MODELS = {
    "base": None,

    # main pipeline
    "stage1": "./sft-lora-phi-2-alpaca-r32-a64-d0.05-lr1.0e-04-wd0.01/final",
    "stage2": "./sft-lora-phi-2-json-r32-a64-d0.05-lr2.0e-05-wd0.01/final",

    # ablation
    "stage2_epoch1": "./sft-lora-phi-2-json-epoch1-r32-a64-d0.05-lr2.0e-05-wd0.01/final",
    "stage2_lr1e5": "./sft-lora-phi-2-json-lr1e5-r32-a64-d0.05-lr1.0e-05-wd0.01/final"
}

DATA_PATH = "json_dataset_final.json"   # JSON eval
MAX_SAMPLES = None

MAX_NEW_TOKENS = 256


# =========================
# LOAD MODEL
# =========================

def load_model(adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


# =========================
# GENERATE
# =========================

def generate(model, tokenizer, instruction, input_text=""):

    sample = {"instruction": instruction, "input": input_text}
    prompt = du.alpaca_row_to_prompt_eval(sample)["text"]

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# =========================
# LOAD DATA
# =========================

def load_json_dataset(path):
    with open(path) as f:
        data = json.load(f)

    return data  


# =========================
# RUN EVAL
# =========================

def run_eval():

    dataset = load_json_dataset(DATA_PATH)

    results = {}

    for model_name, adapter in MODELS.items():

        print(f"\n===== {model_name.upper()} =====")

        model, tokenizer = load_model(adapter)

        model_outputs = []

        for i, item in enumerate(dataset):

            instruction = item["instruction"]
            input_text = item.get("input", "")

            output = generate(model, tokenizer, instruction, input_text)

            print(f"[{i}] done")

            model_outputs.append({
                "instruction": instruction,
                "input": input_text,
                "prediction": output,
                "ground_truth": item["output"]
            })

        results[model_name] = model_outputs

    # save
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Evaluation DONE")


if __name__ == "__main__":
    run_eval()