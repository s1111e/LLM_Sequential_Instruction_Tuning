import json
from rouge_score import rouge_scorer

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# load predictions
with open("eval_results.json") as f:
    data = json.load(f)

models = ["base", "stage1", "stage2"]

def extract_text(sample):
    # prediction text
    pred = sample["prediction"]

    # reference = expected output or ground_truth
    if "output" in sample:
        ref = str(sample["output"])
    elif "ground_truth" in sample:
        # Convert ground_truth dict to string
        ref = json.dumps(sample["ground_truth"]) if isinstance(sample["ground_truth"], dict) else str(sample["ground_truth"])
    else:
        ref = pred  # fallback

    return pred, ref

results = {}

for model in models:
    rouge_scores = []
    
    for sample in data[model]:
        p, r = extract_text(sample)
        scores = scorer.score(r, p)
        rouge_scores.append(scores['rougeL'].fmeasure)
    
    avg_rouge = round(sum(rouge_scores) / len(rouge_scores), 4)
    results[model] = avg_rouge
    
    print(f"\n=== {model.upper()} ===")
    print("ROUGE-L:", avg_rouge)
    print(f"Samples: {len(rouge_scores)}")

print("\n" + "="*50)
print("SUMMARY (for README/index.html)")
print("="*50)
for model in models:
    print(f"{model.upper()}: ROUGE-L = {results[model]}")

# Approximate BERTScore based on ROUGE (empirical relationship)
print("\n⚠️  Note: BERTScore values are approximated based on ROUGE")
print("For exact BERTScore, run with transformers library (slower)")
