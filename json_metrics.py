import json

# =========================
# LOAD DATA
# =========================

with open("eval_results.json") as f:
    data = json.load(f)

models = ["base", "stage1", "stage2", "stage2_epoch1", "stage2_lr1e5"]

# =========================
# HELPERS
# =========================

def is_valid_json(text):
    try:
        json.loads(text)
        return True
    except:
        return False

def normalize_json(obj):
    return json.dumps(obj, sort_keys=True)

# =========================
# METRICS
# =========================

def evaluate_model(outputs):

    total = len(outputs)

    valid = 0
    exact = 0

    for item in outputs:

        pred = item["prediction"]
        gt   = item["ground_truth"]

        # --------------------
        # VALID JSON
        # --------------------
        if is_valid_json(pred):
            valid += 1

            pred_json = json.loads(pred)

            # --------------------
            # EXACT MATCH
            # --------------------
            if normalize_json(pred_json) == normalize_json(gt):
                exact += 1

    return {
        "valid_rate": valid / total,
        "exact_match": exact / total
    }

# =========================
# RUN
# =========================

results = {}

for m in models:
    print(f"\n=== {m.upper()} ===")

    res = evaluate_model(data[m])

    print("JSON validity:", res["valid_rate"])
    print("Exact match :", res["exact_match"])

    results[m] = res

# save
with open("json_metrics_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ JSON metrics DONE")