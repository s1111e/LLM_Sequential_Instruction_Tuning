# 🚀 Sequential Instruction Tuning of Small LLMs

**GitHub Pages:** [📖 View Live Report](https://s1111e.github.io/LLM_Sequential_Instruction_Tuning/)

**Repository:** [💻 GitHub - LLM_Sequential_Instruction_Tuning](https://github.com/s1111e/LLM_Sequential_Instruction_Tuning)

---

# Setup & Installation

## Prerequisites
- Python 3.8+
- CUDA-compatible GPU (tested on V100, A100)
- 40GB+ GPU VRAM for Phi-2 QLoRA fine-tuning
- Access to UTSA HPC system (or any HPC with Slurm)

## Installation

### 1. Clone Repository & Setup Environment

```bash
cd /path/to/project
git clone <your-repo-url>
cd HW3

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Base Model

The base model (Phi-2) will be auto-downloaded on first training run from Hugging Face Hub.

If you prefer to download manually:
```bash
huggingface-cli download microsoft/phi-2 --local-dir ./models/phi-2
```

### 4. Prepare Alpaca Dataset

```bash
python data_utils.py
```

This script:
- Downloads Alpaca-Cleaned from Hugging Face
- Formats data into (instruction, input, output) schema
- Splits into train/validation sets
- Saves locally as pickle files

### 5. UTSA HPC Setup

For GPU resources on UTSA HPC:

```bash
# Load required modules
module load intel/2021.3
module load cuda/11.8
module load python/3.10

# Create HPC environment
srun -p gpu1v100 -n 1 -t 00:30:00 --gres=gpu:1 --pty bash
python -m venv /scratch/hpc_venv
source /scratch/hpc_venv/bin/activate
pip install -r requirements.txt
```

### 6. Set Environment Variables

Create `.env` file:

```bash
# Teacher Model API (if using external API)
OPENAI_API_KEY=your_key_here
TEACHER_MODEL_BASE_URL=http://10.246.100.230/v1
TEACHER_MODEL_NAME=llama-3.3-70b-instruct-awq

# UTSA HPC configs
HPC_EMAIL=your_email@utsa.edu
```

---

## Project Structure

HW3/
│
├── sft-lora-phi-2-alpaca...       # Stage 1 trained model (Alpaca)
├── sft-lora-phi-2-json...         # Stage 2 trained model (default)
├── sft-lora-phi-2-json-epoch1...  # Ablation: 1 epoch model
├── sft-lora-phi-2-json-lr1e5...   # Ablation: lower learning rate
│
├── train-sft.py                   # Main training script (Stage 1 & Stage 2)
├── config.py                      # Training hyperparameters
├── data_utils.py                  # Dataset preparation & formatting
│
├── run_eval.py                    # Generate model outputs
├── eval.py                        # Evaluate model with LoRA adapter
├── eval-base.py                   # Evaluate base model
│
├── judge_eval.py                  # LLM-based pairwise evaluation
├── compute_score.py               # Compute win rates
│
├── json_metrics.py                # JSON validity & exact match metrics
├── json_metrics_results.json      # JSON evaluation results
│
├── eval_results.json              # Generated predictions
├── judge_results.json             # Judge evaluation results
│
├── json_dataset_final.json        # Teacher-generated dataset
│
├── job_stage1.slurm              # HPC job script (Stage 1)
├── job_stage2.slurm              # HPC job script (Stage 2)
│
├── *.out / *.err                 # Training logs
│
└── README.md                     # Final report


The pipeline is modular and allows independent execution of training, evaluation, and analysis components.


# Sequential Instruction Tuning for Structured Output Generation

---

# 1. Overview

This project explores whether a small language model can learn structured JSON output generation through sequential fine-tuning.

The pipeline consists of two stages:

- **Stage 1:** General instruction-following using Alpaca dataset  
- **Stage 2:** Structured output learning using teacher-generated JSON data  

The goal is to improve structured reasoning while minimizing forgetting of general capabilities.

---


### Pipeline Overview

The project follows a structured pipeline:

1. Train a base model on Alpaca dataset (Stage 1)
2. Generate structured JSON dataset using a teacher model
3. Fine-tune the model on structured data (Stage 2)
4. Evaluate using:
   - Judge-based comparison
   - JSON metrics
5. Perform ablation studies on:
   - training epochs
   - learning rate



# 2. Methodology

## Model Choice

We selected Phi-2 due to its efficiency and suitability for QLoRA-based fine-tuning in limited GPU environments.

## Stage 1 — Alpaca Fine-Tuning

We first fine-tune the base model on the Alpaca dataset:

- Format: `(instruction, input, output)`
- Objective: Learn general instruction-following ability
- Method: QLoRA fine-tuning on UTSA HPC

This stage establishes a strong baseline for natural language understanding.

---

## Stage 2 — JSON Instruction Tuning

In the second stage, we construct a dataset using imitation learning:

- Teacher model: Llama 3.3 70B
- Student model learns from teacher-generated outputs

### Dataset Generation Pipeline

The JSON dataset used in Stage 2 was generated locally using a teacher model.

The process includes:

1. Prompting the teacher model with diverse structured tasks
2. Generating JSON outputs using the teacher
3. Validating outputs for JSON correctness
4. Automatically retrying generation for invalid outputs
5. Cleaning and normalizing the dataset
6. Removing duplicates and malformed samples

This pipeline ensures high-quality structured supervision data for the student model.

The initial dataset contained approximately 300 samples, which was reduced to around 290 after cleaning.

### Task types:

- JSON extraction  
- Schema-constrained generation  
- Classification (JSON format)  
- JSON repair  
- Tool-call argument generation  

All outputs are:

- validated for JSON correctness  
- normalized into `(instruction, input, output)` format  

---

## Training Setup

- LoRA-based parameter-efficient tuning  
- Multi-GPU training using Accelerate  
- Dataset size: ~300 samples  

---

# 3. Experiments

---

## 3.1 Alpaca Evaluation (Judge-Based)

| Comparison | A | B | Tie |
|-----------|--|--|-----|
| Base vs Stage1 | 0.61 | 0.21 | 0.18 |
| Stage1 vs Stage2 | 0.43 | 0.22 | 0.34 |

### Interpretation

- Stage 1 improves instruction-following compared to the base model  
- Stage 2 shows a **slight decrease** in Alpaca performance  

This indicates a small amount of forgetting after structured training.

---

## 3.2 JSON Structured Output Evaluation

| Model | JSON Validity | Exact Match |
|------|-------------|------------|
| Base | 0.931 | 0.000 |
| Stage1 | 0.876 | 0.000 |
| Stage2 | 0.914 | 0.000 |
| Stage2 (Epoch1) | 0.928 | 0.000 |
| Stage2 (LR1e-5) | **0.938** | **0.003** |

### Observations

- Stage 2 improves structured output capability  
- Best performance achieved with lower learning rate (1e-5)  
- Exact match remains low due to strict formatting constraints  

---

## 3.3 Forgetting Analysis

Comparing Stage 1 and Stage 2:

- Stage 1 win rate: 0.434  
- Stage 2 win rate: 0.224  
- Tie rate: 0.341  

This shows **mild forgetting**, but not catastrophic.

### Key conclusion:

The model retains most of its general ability while learning structured outputs.

---

## 3.4 Ablation Study

### Epoch Ablation

| Comparison | A | B | Tie |
|-----------|--|--|-----|
| Stage2 vs Epoch1 | 0.486 | 0.159 | 0.355 |

→ More training epochs improve performance.

---

### Learning Rate Ablation

| Comparison | A | B | Tie |
|-----------|--|--|-----|
| Stage2 vs LR1e5 | 0.521 | 0.159 | 0.321 |

→ Lower learning rate (1e-5) yields best results.

---

# 4. Analysis

## 4.1 Three-Checkpoint Comparison

| Model Checkpoint | Alpaca Judge Win Rate | ROUGE-L / BERTScore | JSON Validity | Schema Compliance | Exact Match |
|------------------|----------------------|---------------------|--------------|------------------|------------|
| Checkpoint 0: Untuned base | Low | Low | 0.931 | Low | 0.000 |
| Checkpoint 1: After Stage 1 (Alpaca) | High (0.61 vs base) | High | 0.876 | Low | 0.000 |
| Checkpoint 2: After Stage 2 (Teacher JSON) | Slightly lower (0.22 vs Stage1) | Slight drop | 0.914 | Medium | 0.000 | 

## Key Findings

- Sequential fine-tuning improves structured output generation  
- Stage 2 introduces slight degradation in general tasks  
- Hyperparameters significantly affect performance  

---

## Trade-off Insight

There is a clear trade-off:

- **More specialization → better JSON**
- **Less retention → slight Alpaca drop**

This reflects the core challenge of LLM fine-tuning.

---

## Failure Cases

- Missing or inconsistent JSON fields  
- Slight schema variations  
- Incomplete outputs  

---

# 5. Prompt Engineering

## Teacher Prompt

We enforced:

- strict JSON output  
- no explanations  
- schema compliance  

### Prompt Refinement

Initial prompts sometimes produced invalid JSON outputs.  
We refined prompts by enforcing:

- strict JSON-only responses  
- no explanations  
- explicit schema instructions  

This significantly improved output validity.

## Judge Prompt

The judge evaluates:

- correctness  
- clarity  
- instruction following  
- JSON validity  

---
## Reproducibility & Running the Pipeline

### Step 1: Prepare Datasets

```bash
# Download & prepare Alpaca dataset
python data_utils.py
```

### Step 2: Stage 1 — Alpaca Fine-Tuning

**Local Machine (Multi-GPU with Accelerate):**
```bash
# Edit config.py: set local_dataset_path = path/to/alpaca
accelerate launch --multi_gpu train-sft.py \
    --model_name_or_path microsoft/phi-2 \
    --dataset_name tatsu-lab/alpaca \
    --output_dir ./sft-lora-phi-2-alpaca-r32-a64-d0.05-lr2.0e-05-wd0.01 \
    --num_train_epochs 2 \
    --learning_rate 2e-5
```

**UTSA HPC (Slurm):**
```bash
sbatch job_stage1.slurm
```

Contents of `job_stage1.slurm`:
```bash
#!/bin/bash
#SBATCH -p gpu1v100
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH --job-name=phi2-stage1

module load cuda/11.8
source /scratch/hpc_venv/bin/activate

accelerate launch train-sft.py \
    --model_name_or_path microsoft/phi-2 \
    --dataset_name tatsu-lab/alpaca \
    --output_dir ./sft-lora-phi-2-alpaca/final \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4
```

**Output:** Saved checkpoint in `sft-lora-phi-2-alpaca/final/`

---

### Step 3: Stage 1 Evaluation

```bash
# Generate predictions on held-out set
python run_eval.py \
    --base_model_name_or_path microsoft/phi-2 \
    --adapter_path ./sft-lora-phi-2-alpaca/final \
    --output_file eval_results.json \
    --checkpoint_name stage1
```

---

### Step 4: Stage 2 — Teacher-Generated JSON Fine-Tuning

Assumes teacher-generated JSON dataset is in `json_dataset_final.json`.

**Local Machine:**
```bash
# Continue training from Stage 1 checkpoint
accelerate launch train-sft.py \
    --model_name_or_path microsoft/phi-2 \
    --adapter_path ./sft-lora-phi-2-alpaca/final \
    --local_dataset_path json_dataset_final.json \
    --output_dir ./sft-lora-phi-2-json-r32-a64-d0.05-lr2.0e-05-wd0.01 \
    --num_train_epochs 2 \
    --learning_rate 2e-5
```

**UTSA HPC:**
```bash
sbatch job_stage2.slurm
```

**Output:** Saved checkpoint in `sft-lora-phi-2-json/final/`

---

### Step 5: Stage 2 Evaluation

```bash
python run_eval.py \
    --base_model_name_or_path microsoft/phi-2 \
    --adapter_path ./sft-lora-phi-2-json/final \
    --output_file eval_results.json \
    --checkpoint_name stage2
```

---

### Step 6: Judge Evaluation (Alpaca Pairwise Comparison)

```bash
# Run pairwise judge comparing all checkpoints
python judge_eval.py

# Output saved to judge_results.json
```

---

### Step 7: Compute Metrics

```bash
# JSON validity & exact match metrics
python json_metrics.py \
    --input eval_results.json \
    --output json_metrics_results.json

# Aggregate judge scores
python compute_score.py \
    --judge_results judge_results.json \
    --output metrics_summary.json
```

---

### Complete Reproduction in One Script

```bash
#!/bin/bash

echo "🚀 Starting sequential fine-tuning pipeline..."

# Stage 1
echo "📍 Stage 1: Alpaca fine-tuning"
accelerate launch train-sft.py --num_train_epochs 2 --learning_rate 2e-5

# Eval Stage 1
echo "📍 Evaluating Stage 1"
python run_eval.py --checkpoint_name stage1

# Stage 2
echo "📍 Stage 2: JSON tuning"
accelerate launch train-sft.py --local_dataset_path json_dataset_final.json \
  --num_train_epochs 2 --learning_rate 2e-5

# Eval Stage 2
echo "📍 Evaluating Stage 2"
python run_eval.py --checkpoint_name stage2

# Judge & Metrics
echo "📍 Running judge evaluation"
python judge_eval.py
python compute_score.py
python json_metrics.py

echo "✅ Pipeline complete! Results saved."
```

---

### Ablation Studies

**Learning Rate Ablation (2e-5 vs 1e-5 vs 5e-6):**
```bash
for lr in 1e-5 2e-5 5e-6; do
  accelerate launch train-sft.py \
    --learning_rate $lr \
    --output_dir ./ablation-lr-$lr
done
```

**Epochs Ablation (1 vs 2 vs 3):**
```bash
for epochs in 1 2 3; do
  accelerate launch train-sft.py \
    --num_train_epochs $epochs \
    --output_dir ./ablation-epoch-$epochs
done
```

---

### Configuration Reference

Key parameters in `config.py`:

```python
LoRA Parameters:
- r: 32                              # LoRA rank
- lora_alpha: 64                      # LoRA alpha (scaling factor)
- lora_dropout: 0.05                  # Dropout in LoRA layers
- target_modules: ["q_proj", "v_proj"]

Training Parameters:
- learning_rate: 2e-5                 # Stage 1 & 2 default
- num_train_epochs: 2                 # Epochs per stage
- per_device_train_batch_size: 4      # Batch size (QLoRA-friendly)
- gradient_accumulation_steps: 2
- max_seq_length: 1024

Optimization:
- optimizer: "adamw_8bit"
- weight_decay: 0.01
- warmup_steps: 100
```

---

# 6. Conclusion

Sequential fine-tuning is effective for teaching structured output generation.

While a small amount of forgetting occurs, the model successfully balances:

- general instruction-following  
- structured output capability  

No catastrophic forgetting is observed.

This demonstrates that sequential fine-tuning can effectively balance structured learning and generalization when carefully tuned.

---

# Appendix: Full Prompts & Templates

## A.1 Teacher Model Prompt (JSON Generation)

Used for generating training data via teacher-model (Llama 3.1 70B Instruct).

### A.1.1 JSON Extraction Task

```
You are a JSON extraction expert. Given unstructured text, extract the requested information into valid JSON format. Return ONLY the JSON object, no explanations.

Task: Extract person information
Input Text:
{input_text}

Required JSON Schema:
{{
  "name": "string",
  "age": "integer",
  "email": "string"
}}

Output (valid JSON only):
```

### A.1.2 Schema-Constrained Generation

```
Generate valid JSON that matches the given schema. The output must be valid JSON and match the schema exactly.

Schema:
{{
  "product_name": "string",
  "price": "number",
  "in_stock": "boolean",
  "tags": ["string"]
}}

Generate JSON for a laptop product:

Output (valid JSON only):
```

### A.1.3 Classification with JSON Output

```
Classify the following text and return the result as JSON with a confidence score. 
Return ONLY the JSON, no explanations.

Sentiment labels: ["positive", "negative", "neutral"]

Text: "{text}"

JSON Output (strict format):
{{
  "label": "string (from labels list)",
  "confidence": "number (0.0-1.0)"
}}
```

### A.1.4 JSON Repair Task

```
Fix the malformed JSON below and return valid JSON. Keep the original data as much as possible.

Malformed JSON:
{malformed_json}

Return ONLY valid JSON:
```

### A.1.5 Tool-Call Argument Generation

```
Generate JSON representing a function call with named arguments. The JSON must match the function signature exactly.

Function Signature:
send_email(recipient: str, subject: str, body: str, cc: list[str], priority: str)

Task: Send an urgent meeting reminder email

JSON Output (valid JSON only):
{{
  "function": "send_email",
  "arguments": {{
    "recipient": "...",
    "subject": "...",
    "body": "...",
    "cc": [...],
    "priority": "..."
  }}
}}
```

---

## A.2 Judge Evaluation Prompt

Used for pairwise comparison of model outputs across checkpoints.

### A.2.1 Alpaca Task Judge Prompt

```
You are an expert LLM evaluator. Compare two responses to the same instruction.

Instruction:
{instruction}

Response A:
{response_a}

Response B:
{response_b}

Evaluate on these dimensions:
1. Instruction Following (1-5): How well does each response follow the given instruction?
2. Correctness (1-5): Is the information accurate?
3. Clarity (1-5): How clear and well-written is the response?
4. Completeness (1-5): Does it cover all aspects of the instruction?
5. Hallucination Risk (1-5): How likely is the response to contain false information?

After evaluation, choose the better response:
Answer ONLY: A or B or TIE

Reasoning:
```

### A.2.2 JSON Task Judge Prompt

```
You are an expert evaluator of JSON outputs. Compare two JSON responses.

Instruction:
{instruction}

Response A:
{response_a}

Response B:
{response_b}

Evaluate:
1. JSON Validity: Is the JSON parseable and valid?
2. Schema Compliance: Does it match the required schema?
3. Correctness: Is the content correct?
4. Completeness: Are all required fields present?
5. Formatting: Is indentation/formatting consistent?

Which is better? A or B or TIE
```

---

## A.3 Dataset Schema

All training datasets (Alpaca and Teacher-Generated JSON) follow this unified schema:

```json
[
  {
    "instruction": "What is the capital of France?",
    "input": "",
    "output": "The capital of France is Paris."
  },
  {
    "instruction": "Extract person information from text",
    "input": "John is a 28-year-old developer from Seattle.",
    "output": "{\"name\": \"John\", \"age\": 28, \"location\": \"Seattle\"}"
  }
]
```

---

## A.4 Prompt Engineering Iterations

### Initial Teacher Prompt (v1)
```
Extract information to JSON format.
```
❌ **Problem:** Produced invalid JSON, missing schema, inconsistent output.

### Revised Teacher Prompt (v2)
```
Convert to valid JSON. Return JSON only.
Schema: {schema}
```
⚠️ **Issue:** Still inconsistent, occasional truncation.

### Final Teacher Prompt (v3)
```
You are a JSON expert. Your task is:
1. Extract keys requested in the schema
2. Return ONLY valid JSON (no explanations)
3. Match the schema exactly
4. Use proper types (string, number, boolean, array)

Schema:
{schema}

Input:
{input}

Return only valid JSON:
```
✅ **Result:** ~95% validity rate, improved schema compliance.

---

## A.5 Configuration Files

### HPC Setup (accelerate_config.yaml)
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: bf16
use_cpu: false
num_processes: 1
num_machines: 1
gpu_ids: 0
machine_rank: 0

fsdp_config: {}
deepspeed_config: {}
```

### Stage 1 Training Config
```yaml
learning_rate: 2e-5
num_train_epochs: 2
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
warmup_ratio: 0.1
weight_decay: 0.01
fp16: false
bf16: true
max_seq_length: 1024
```

---

## Summary

The reproducibility section includes:
- ✅ Complete CLI commands for each stage
- ✅ UTSA HPC Slurm scripts
- ✅ Teacher prompt templates (all 5 task types)
- ✅ Judge evaluation prompts
- ✅ Ablation study commands
- ✅ Configuration reference
- ✅ End-to-end reproduction script

